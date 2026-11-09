"""Tree-structured prior, after TabICL's tree-SCM, TabForestPFN's forest generator (MIT) and Mitra's DSRF, re-implemented
without sklearn: random axis-aligned trees with random thresholds and random leaf values, summed as a small random forest
or boosted stack, applied to Gaussian (or mixed) inputs. Brings the piecewise-constant, interaction-rich structure that
the smooth MLP prior under-represents."""
from __future__ import annotations

import numpy as np

from .contract import finalize, quantise, reg2cls_binary, tnlu


def random_tree(rng, X: np.ndarray, depth: int, feature_subset: np.ndarray):
    """Returns leaf values per row for one random tree grown on X: random feature and a threshold drawn from the data at each node."""
    n = X.shape[0]; out = np.zeros(n); idx = [np.arange(n)]
    for level in range(depth):
        new = []
        for rows in idx:
            if len(rows) < 4: new.append(rows); continue
            j = rng.choice(feature_subset); col = X[rows, j]; thr = rng.choice(col)
            left = rows[col <= thr]; right = rows[col > thr]
            if len(left) == 0 or len(right) == 0: new.append(rows); continue
            new += [left, right]
        idx = new
    for rows in idx: out[rows] = rng.standard_normal()
    return out


class TreeSCM:
    def __init__(self, rng, d: int):
        self.rng, self.d = rng, d
        self.n_trees = int(np.clip(round(tnlu(rng, 1, 40)), 1, 40)); self.depth = int(np.clip(round(1 + rng.exponential(2)), 1, 6))
        self.boosted = rng.random() < 0.4; self.noise = float(tnlu(rng, 0.01, 0.5))
        self.n_causes = int(rng.integers(max(1, d // 2), d + 1)); self.heavy_tail = rng.random() < 0.3
        self.feature_subset = rng.choice(d, max(1, int(np.ceil(np.sqrt(d)))), replace=False) if rng.random() < 0.5 else np.arange(d)

    def forward(self, n: int, shift: float = 0.0):
        rng = self.rng
        X = rng.standard_normal((n, self.d)) if not self.heavy_tail else rng.standard_t(3, (n, self.d))
        # a few features are deterministic functions of others (collinear structure)
        for j in range(self.n_causes, self.d):
            a, b = rng.choice(self.n_causes, 2) if self.n_causes >= 2 else (0, 0); X[:, j] = np.tanh(X[:, a] + 0.5 * X[:, b]) + 0.3 * rng.standard_normal(n)
        target = np.zeros(n); Xt = X + shift
        for t in range(self.n_trees):
            target += random_tree(rng, Xt, self.depth, self.feature_subset) * (0.7 ** t if self.boosted else 1.0)
        target = (target - target.mean()) / (target.std() + 1e-6) + self.noise * rng.standard_normal(n)
        return X, target


def sample(rng, n=None, d=None, prevalence=None, drift=None, finalize_table=True, fin=None):
    n = n or int(rng.integers(256, 1025)); d = d or int(np.clip(round(tnlu(rng, 2, 64)), 2, 64))
    scm = TreeSCM(rng, d)
    if drift is not None:
        X, target, cohort = drift(scm, n, rng)
    else:
        X, target = scm.forward(n); cohort = None
    if prevalence is None and rng.random() < 0.5: prevalence = float(rng.uniform(0.02, 0.5))
    y = reg2cls_binary(rng, target, prevalence); is_cat = np.zeros(d, bool)
    if rng.random() < 0.3:
        for j in rng.choice(d, int(rng.integers(1, max(2, d // 2 + 1))), replace=False):
            X[:, j] = quantise(rng, X[:, j], int(min(12, max(2, round(rng.gamma(1, 10)))))); is_cat[j] = True
    meta = dict(component="tree_scm", n_trees=scm.n_trees, depth=scm.depth, boosted=scm.boosted, prevalence_target=prevalence)
    if not finalize_table: return X.astype(np.float32), y, is_cat, cohort, meta
    return finalize(rng, X, y, is_cat, cohort, meta, **(fin or {}))
