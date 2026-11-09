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


def ks(y, score) -> float:
    """Kolmogorov-Smirnov statistic: largest gap between the score distributions of the two classes."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(score)); return float(np.max(tpr - fpr))


def reliability_bins(y, p, n_bins=10):
    """Equal-count bins of the predicted probability: mean prediction, observed rate and count per bin."""
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64); order = np.argsort(p, kind="stable")
    parts = [c for c in np.array_split(order, n_bins) if len(c)]
    return np.array([p[c].mean() for c in parts]), np.array([y[c].mean() for c in parts]), np.array([len(c) for c in parts])


def ece(y, p, n_bins=10) -> float:
    """Expected calibration error over equal-count bins: count-weighted mean of |observed rate - mean prediction|."""
    mp, obs, cnt = reliability_bins(y, p, n_bins); return float(np.sum(cnt * np.abs(obs - mp)) / cnt.sum())


def calibration(y, score) -> dict:
    """Probability-quality metrics for a log-odds score. brier_skill is 1 - Brier / Brier of the constant prevalence forecast
    (0 means no better than quoting the base rate, 1 is perfect)."""
    y = np.asarray(y); p = sigmoid(score); prev = float(y.mean()); brier = float(np.mean((p - y) ** 2))
    return {"auc": float(roc_auc_score(y, score)), "ks": ks(y, score), "brier": brier, "brier_skill": float(1 - brier / (prev * (1 - prev))),
            "ece": ece(y, p), "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "mean_pred": float(p.mean()), "prevalence": prev}
