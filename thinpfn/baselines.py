"""Reference scorers for the small-n curves and the gates: logistic regression, XGBoost (Qwenderwriter grid, picked on
a held-out part of the context), TabPFN v2 (installed weights, GPU). Every scorer has the same signature:
fit on (X_ctx, y_ctx), return scores for X_query. NaNs are allowed in X (XGBoost and TabPFN handle them; logistic
regression gets median imputation)."""
from __future__ import annotations

import numpy as np

XGB_GRID = [
    {"max_depth": 3, "learning_rate": 0.05, "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 3},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 5, "reg_alpha": 0.1},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 8, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 8, "reg_alpha": 0.2},
]
SMALL_N_GRID = [{"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 2, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 3}]


def logistic(X_ctx, y_ctx, X_q, seed=0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
    m.fit(X_ctx, y_ctx); return m.decision_function(X_q)


def xgboost(X_ctx, y_ctx, X_q, seed=0):
    """Grid picked by log loss on a stratified 20 percent hold-out of the context (with early stopping), then refit on all
    the context with the chosen round count. At n < 300 the hold-out is too small to select, so a fixed shallow config is used."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    n = len(y_ctx); grid = XGB_GRID if n >= 300 else SMALL_N_GRID
    base = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", seed=seed, nthread=8)
    if n >= 300:
        Xa, Xb, ya, yb = train_test_split(X_ctx, y_ctx, test_size=0.2, random_state=seed, stratify=y_ctx)
        best = None
        for g in grid:
            b = xgb.train({**base, **g}, xgb.DMatrix(Xa, ya), 2000, evals=[(xgb.DMatrix(Xb, yb), "v")], early_stopping_rounds=50, verbose_eval=False)
            if best is None or b.best_score < best[0]: best = (b.best_score, g, b.best_iteration + 1)
        _, g, rounds = best
    else:
        g, rounds = grid[0], max(20, min(200, n // 2))
    b = xgb.train({**base, **g}, xgb.DMatrix(X_ctx, y_ctx), rounds)
    return b.predict(xgb.DMatrix(X_q), output_margin=True)


_TABPFN = {}


def tabpfn_v2(X_ctx, y_ctx, X_q, seed=0, device=None):
    """TabPFN v2 classifier (tabpfn 2.0.9 in the released benchmark), library defaults (4 estimators in 2.0.9), weights downloaded by the library on first use. device "cuda" or "cpu"."""
    from tabpfn import TabPFNClassifier
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu"); key = (device, seed)
    if key not in _TABPFN:
        _TABPFN[key] = TabPFNClassifier(device=device, random_state=seed, ignore_pretraining_limits=True)
    m = _TABPFN[key]; m.fit(X_ctx, y_ctx)
    p = m.predict_proba(X_q)[:, 1]; p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


BASELINES = {"logistic": logistic, "xgboost": xgboost, "tabpfn_v2": tabpfn_v2}
