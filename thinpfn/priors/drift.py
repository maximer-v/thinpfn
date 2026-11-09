"""Drift wrapper for the generic components (after Drift-Resilient TabPFN's second-order SCM, re-implemented): each row
gets a cohort index in [0, 1]; the generating function's parameters move along the cohort (a sparse subset of weights
for the MLP-SCM, an input shift for the tree-SCM), and so can the label boundary (prevalence drift). The cohort is
emitted so finalize() can build an out-of-time context / query split and the model can read it through its domain
channel. Implemented by generating the rows in K cohort buckets with interpolated parameters."""
from __future__ import annotations

import numpy as np

from .contract import tnlu


def make_drift(rng, n_buckets: int = 8):
    magnitude = float(tnlu(rng, 0.05, 1.0)); sparsity = float(rng.uniform(0.05, 0.4)); smooth = rng.random() < 0.7

    def apply(scm, n, rng):
        cohort = rng.random(n); edges = np.linspace(0, 1, n_buckets + 1); bucket = np.clip(np.digitize(cohort, edges) - 1, 0, n_buckets - 1)
        Xs, ts = np.zeros((n, scm.d)), np.zeros(n)
        if hasattr(scm, "W"):  # MLP-SCM: perturb a sparse subset of weights
            masks = [(rng.random(W.shape) < sparsity) for W in scm.W]; dirs = [rng.standard_normal(W.shape) * magnitude * np.abs(W).mean() for W in scm.W]
            if not smooth: flip = rng.uniform(0.3, 0.7)
            for b in range(n_buckets):
                rows = np.flatnonzero(bucket == b)
                if len(rows) == 0: continue
                c = (b + 0.5) / n_buckets; t = c if smooth else float(c > flip)
                delta = [m * d * t for m, d in zip(masks, dirs)]
                X, target = scm.forward(len(rows), weight_delta=delta); Xs[rows] = X; ts[rows] = target
        else:  # tree-SCM: shift inputs along the cohort
            for b in range(n_buckets):
                rows = np.flatnonzero(bucket == b)
                if len(rows) == 0: continue
                c = (b + 0.5) / n_buckets; shift = magnitude * (c - 0.5) if smooth else magnitude * (float(c > 0.5) - 0.5)
                X, target = scm.forward(len(rows), shift=shift); Xs[rows] = X; ts[rows] = target
        # prevalence drift: tilt the continuous target along the cohort
        if rng.random() < 0.5: ts = ts + magnitude * (cohort - 0.5) * ts.std()
        return Xs, ts, cohort.astype(np.float32)

    apply.meta = dict(drift_magnitude=magnitude, drift_sparsity=sparsity, drift_smooth=smooth)
    return apply
