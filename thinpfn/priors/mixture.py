"""The prior mixture: draws a component by weight, applies the drift wrapper to the generic components with a probability,
runs the learnability filter and returns a finalized Table. Also a torch-free batch collate for the trainer."""
from __future__ import annotations

import numpy as np

from . import credit_ledger, mlp_scm, tree_scm
from .contract import Table
from .drift import make_drift

# The released model (thinpfn_v1) was trained on DEFAULT: two generic structural-causal priors, the credit ledger simulator,
# and drifting variants of the generic priors (a cohort index moves the generating function).
DEFAULT_WEIGHTS = {"mlp_scm": 0.40, "tree_scm": 0.20, "credit_ledger": 0.30, "mlp_scm_drift": 0.07, "tree_scm_drift": 0.03}
GENERIC_ONLY = {"mlp_scm": 0.65, "tree_scm": 0.35}
GENERIC_DRIFT = {"mlp_scm": 0.50, "tree_scm": 0.25, "mlp_scm_drift": 0.17, "tree_scm_drift": 0.08}
LEDGER_ONLY = {"credit_ledger": 1.0}
MIXTURES = {"default": DEFAULT_WEIGHTS, "generic": GENERIC_ONLY, "generic_drift": GENERIC_DRIFT, "ledger": LEDGER_ONLY}


def learnability(table: Table, rng, lo: float = 0.55, hi: float = 0.995) -> tuple[bool, float]:
    """ExtraTrees fit on the context, AUC on the query rows; tables that are unpredictable or trivial are dropped."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import roc_auc_score
    X = np.nan_to_num(table.X, nan=0.0); yq = table.y[~table.context]
    if yq.min() == yq.max(): return True, float("nan")       # cannot be judged; keep
    m = ExtraTreesClassifier(n_estimators=40, max_features="sqrt", min_samples_leaf=2, random_state=int(rng.integers(1 << 31)), n_jobs=1)
    m.fit(X[table.context], table.y[table.context]); auc = float(roc_auc_score(yq, m.predict_proba(X[~table.context])[:, 1]))
    return lo <= auc <= hi, auc


class PriorMixture:
    def __init__(self, weights: dict | str = "default", n_range=(256, 1024), d_max=64, filter_tables=True, seed=0, targets: dict | None = None):
        self.weights = MIXTURES[weights] if isinstance(weights, str) else weights
        self.targets = targets
        self.names = list(self.weights); self.p = np.array([self.weights[k] for k in self.names]); self.p /= self.p.sum()
        self.n_range, self.d_max, self.filter = n_range, d_max, filter_tables
        self.rng = np.random.default_rng(seed); self.stats = {"drawn": 0, "dropped": 0}

    def reseed(self, seed: int): self.rng = np.random.default_rng(seed)

    def one(self, n: int | None = None) -> Table:
        rng = self.rng
        for _ in range(50):
            name = rng.choice(self.names, p=self.p); n_ = n or int(rng.integers(self.n_range[0], self.n_range[1] + 1))
            try:
                if name == "credit_ledger": t = credit_ledger.sample(rng, n=n_, targets=self.targets)
                elif name == "mlp_scm": t = mlp_scm.sample(rng, n=n_)
                elif name == "tree_scm": t = tree_scm.sample(rng, n=n_)
                elif name == "mlp_scm_drift": t = mlp_scm.sample(rng, n=n_, drift=make_drift(rng))
                elif name == "tree_scm_drift": t = tree_scm.sample(rng, n=n_, drift=make_drift(rng))
                else: raise ValueError(name)
            except (AssertionError, np.linalg.LinAlgError, ValueError):
                self.stats["dropped"] += 1; continue
            if t.d > self.d_max: keep = np.sort(rng.choice(t.d, self.d_max, replace=False)); t = Table(t.X[:, keep], t.y, t.is_categorical[keep], t.cohort, t.context, t.meta, None if t.tags is None else t.tags[keep])
            self.stats["drawn"] += 1
            if self.filter:
                ok, auc = learnability(t, rng); t.meta["filter_auc"] = auc
                if not ok: self.stats["dropped"] += 1; continue
            t.meta["mixture_component"] = str(name)
            return t
        raise RuntimeError("prior mixture failed to produce a table in 50 attempts")

    def batch(self, size: int, n: int | None = None) -> list[Table]:
        n_ = n or int(self.rng.integers(self.n_range[0], self.n_range[1] + 1))   # one row count per batch so tables stack
        return [self.one(n_) for _ in range(size)]


def collate(tables: list[Table], d_max: int = 64):
    """Stack a batch of tables of equal n into padded arrays: X [B, n, d_max] (NaN kept, padding columns NaN), feature mask
    [B, d_max], y [B, n], context [B, n], cohort [B, n], is_categorical [B, d_max], tags [B, d_max]."""
    B = len(tables); n = tables[0].n
    X = np.full((B, n, d_max), np.nan, np.float32); fm = np.zeros((B, d_max), bool); cat = np.zeros((B, d_max), bool)
    y = np.zeros((B, n), np.int64); ctx = np.zeros((B, n), bool); coh = np.zeros((B, n), np.float32); tags = np.zeros((B, d_max), np.int64)
    for i, t in enumerate(tables):
        assert t.n == n, "tables in a batch must share the row count"
        X[i, :, :t.d] = t.X; fm[i, :t.d] = True; cat[i, :t.d] = t.is_categorical; y[i] = t.y; ctx[i] = t.context; coh[i] = t.cohort
        if t.tags is not None: tags[i, :t.d] = t.tags
    return {"X": X, "feature_mask": fm, "is_categorical": cat, "y": y, "context": ctx, "cohort": coh, "tags": tags}
