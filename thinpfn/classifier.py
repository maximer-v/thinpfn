"""scikit-learn style front end. `fit` stores the labelled rows (the context); `predict_proba` scores new rows in one forward
pass of the pretrained network, with no gradient steps. Numeric features (NaN allowed); pandas frames are accepted and
string, category and boolean columns are ordinal-encoded with a mapping learned on the context.

    from thinpfn import ThinPFNClassifier
    clf = ThinPFNClassifier().fit(X_train, y_train)       # 200 to a few thousand labelled rows
    p = clf.predict_proba(X_new)[:, 1]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from .model import PFNConfig, ThinPFN

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
DEFAULT_WEIGHTS = "thinpfn_v1"
_MODEL_PARAMS = ("weights", "device")


def load_pretrained(name_or_path: str | Path = DEFAULT_WEIGHTS, device: str | torch.device | None = None) -> ThinPFN:
    """Load released weights (thinpfn/weights/<name>.safetensors + <name>.json) or a training checkpoint (.pt with 'model' and 'config')."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    p = Path(name_or_path)
    if not p.exists(): p = WEIGHTS_DIR / f"{name_or_path}.safetensors"
    if not p.exists(): raise FileNotFoundError(f"no weights at {name_or_path} (released names: {[q.stem for q in WEIGHTS_DIR.glob('*.safetensors')]})")
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        cfg = PFNConfig(**json.load(open(p.with_suffix(".json")))["config"]); state = load_file(str(p))
    else:
        st = torch.load(p, map_location="cpu"); cfg = PFNConfig(**st["config"]); state = st["model"]
    model = ThinPFN(cfg); model.load_state_dict({k: v.float() for k, v in state.items()}); return model.to(device).eval()


def _is_text_like(col) -> bool:
    import pandas as pd
    dt = col.dtype
    return dt == object or dt == bool or isinstance(dt, (pd.CategoricalDtype, pd.StringDtype, pd.BooleanDtype)) or str(dt) in ("string", "str", "boolean")


class ThinPFNClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier backed by the pretrained ThinPFN network.

    weights: released weight name or a path. device: "cpu", "cuda" or None (cuda if available). n_perm: size of the column-
    permutation ensemble (8 recommended, 0 for one pass). preprocess: "qnorm" (recommended) or "z". categorical: None (all
    columns numeric, the convention the released weights were trained with), "auto" (flag columns with <= 12 distinct integer
    values) or a boolean mask of length n_features. max_context: stratified subsample of the context above this many rows.
    seed: for the permutations and the context subsample."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device=None, n_perm: int = 8, preprocess: str = "qnorm", categorical=None, max_context: int = 4096, seed: int = 0):
        self.weights = weights; self.device = device; self.n_perm = n_perm; self.preprocess = preprocess; self.categorical = categorical; self.max_context = max_context; self.seed = seed

    # ---------------------------------------------------------------------------------------------------- model cache
    @property
    def model(self) -> ThinPFN:
        key = (str(self.weights), str(self.device))
        if getattr(self, "_model_key", None) != key: self._model = load_pretrained(self.weights, self.device); self._model_key = key
        return self._model

    def set_params(self, **params):
        unknown = set(params) - set(self.get_params())
        if unknown: raise ValueError(f"unknown parameters {sorted(unknown)}; valid: {sorted(self.get_params())}")
        if any(k in _MODEL_PARAMS for k in params): self._model_key = None
        return super().set_params(**params)

    def __getstate__(self):   # the network is not pickled with the estimator; it reloads from `weights` on first use
        d = self.__dict__.copy(); d.pop("_model", None); d["_model_key"] = None; return d

    # ---------------------------------------------------------------------------------------------------- encoding
    def _encode(self, X, fit: bool) -> np.ndarray:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            if fit: self.feature_names_in_ = np.array(list(X.columns), dtype=object); self.encoders_ = {c: None for c in X.columns}
            missing = [c for c in self.feature_names_in_ if c not in X.columns]
            if missing: raise ValueError(f"columns missing at predict time: {missing}")
            X = X[list(self.feature_names_in_)].copy()
            for c in X.columns:
                if fit and _is_text_like(X[c]):
                    vals = X[c].dropna().astype(str); self.encoders_[c] = {v: i for i, v in enumerate(sorted(vals.unique()))}
                enc = self.encoders_.get(c)
                if enc is not None:   # a column encoded at fit time is encoded with the same mapping, whatever dtype arrives now
                    X[c] = X[c].astype(object).where(X[c].notna(), None).map(lambda v: enc.get(str(v), np.nan) if v is not None else np.nan).astype(float)
                else:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
            X = X.to_numpy(dtype=np.float32)
        else:
            if fit: self.encoders_ = {}
        X = np.array(X, dtype=np.float32, copy=True)
        if X.ndim != 2: raise ValueError("X must be 2-D")
        return X

    # ---------------------------------------------------------------------------------------------------- sklearn API
    def fit(self, X, y):
        X = self._encode(X, fit=True); y = np.asarray(y)
        if len(y) != len(X): raise ValueError(f"X has {len(X)} rows and y has {len(y)}")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2: raise ValueError(f"ThinPFN is a binary classifier; got classes {self.classes_}")
        if X.shape[1] > self.model.cfg.max_features: raise ValueError(f"at most {self.model.cfg.max_features} features are supported (got {X.shape[1]}); select or project features first")
        if self.max_context < 2: raise ValueError("max_context must be at least 2")
        if self.categorical is not None and not isinstance(self.categorical, str):
            mask = np.asarray(self.categorical, bool)
            if mask.shape != (X.shape[1],): raise ValueError(f"categorical mask must have length {X.shape[1]}")
        elif isinstance(self.categorical, str) and self.categorical != "auto": raise ValueError('categorical must be None, "auto" or a boolean mask')
        yb = (y == self.classes_[1]).astype(np.int64)
        if len(yb) > self.max_context:   # stratified subsample of exactly max_context rows, both classes kept
            rng = np.random.default_rng(self.seed); pos, neg = np.flatnonzero(yb == 1), np.flatnonzero(yb == 0)
            n_pos = int(np.clip(round(self.max_context * len(pos) / len(yb)), 1, self.max_context - 1))
            keep = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, self.max_context - n_pos, replace=False)])); X, yb = X[keep], yb[keep]
        self.X_, self.y_ = X, yb; self.n_features_in_ = X.shape[1]
        return self

    def _check_fitted(self):
        if not hasattr(self, "X_"): raise RuntimeError("call fit(X, y) before predicting")

    def decision_function(self, X):
        self._check_fitted(); X = self._encode(X, fit=False)
        if X.shape[1] != self.n_features_in_: raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        cat = False if self.categorical is None else (None if isinstance(self.categorical, str) else np.asarray(self.categorical, bool))
        return self.model.predict(self.X_, self.y_, X, is_cat=cat, preprocess=self.preprocess, n_perm=self.n_perm, seed=self.seed)

    def predict_proba(self, X):
        from scipy.special import expit
        p1 = expit(self.decision_function(X)); return np.stack([1 - p1, p1], 1)

    def predict(self, X):
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def _more_tags(self):   # scikit-learn < 1.6
        return {"allow_nan": True, "binary_only": True}

    def __sklearn_tags__(self):   # scikit-learn >= 1.6
        tags = super().__sklearn_tags__(); tags.input_tags.allow_nan = True; tags.classifier_tags.multi_class = False; return tags
