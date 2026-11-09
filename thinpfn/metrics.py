from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=np.float64)))


def predictive(y, score) -> dict:
    y = np.asarray(y); p = np.clip(sigmoid(score), 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y, score)), "log_loss": float(log_loss(y, p)), "n": int(len(y)), "prevalence": float(y.mean())}


def paired_bootstrap(y, s_a, s_b, n_boot=2000, seed=0) -> dict:
    """AUC(a) - AUC(b) on the same rows, bootstrap SE and 95 percent CI."""
    y = np.asarray(y); s_a = np.asarray(s_a); s_b = np.asarray(s_b); rng = np.random.default_rng(seed)
    d0 = roc_auc_score(y, s_a) - roc_auc_score(y, s_b); ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        ds.append(roc_auc_score(y[i], s_a[i]) - roc_auc_score(y[i], s_b[i]))
    ds = np.asarray(ds)
    return {"delta": float(d0), "se": float(ds.std()), "ci95": [float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))], "significant": bool(np.quantile(ds, 0.025) > 0 or np.quantile(ds, 0.975) < 0)}
