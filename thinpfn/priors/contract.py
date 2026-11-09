"""The common output contract of every prior component and the shared post-processing.

A Table is one synthetic dataset: X [n, d] float32 with NaN allowed, y [n] int64 in {0, 1}, is_categorical [d] bool,
cohort [n] float32 in [0, 1] (0 everywhere for i.i.d. tables), context [n] bool (True = labelled context row).
finalize() applies what every component shares: optional uninformative / collinear / ratio columns, missingness,
the context split (random or out-of-time along the cohort), z-scoring on CONTEXT rows only with clipping, and random
row and column permutations."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Table:
    X: np.ndarray
    y: np.ndarray
    is_categorical: np.ndarray
    cohort: np.ndarray
    context: np.ndarray
    meta: dict = field(default_factory=dict)
    tags: np.ndarray | None = None      # [d] credit field type ids (schema.py); None = all UNKNOWN

    @property
    def n(self): return self.X.shape[0]
    @property
    def d(self): return self.X.shape[1]

    def check(self):
        n, d = self.X.shape
        assert self.X.dtype == np.float32 and self.y.shape == (n,) and self.is_categorical.shape == (d,) and self.cohort.shape == (n,) and self.context.shape == (n,)
        assert set(np.unique(self.y)) <= {0, 1}, "binary labels expected"
        assert self.context.sum() >= 2 and (~self.context).sum() >= 1, "need context and query rows"
        yc = self.y[self.context]; assert yc.min() != yc.max(), "context must contain both classes"
        assert np.isfinite(self.cohort).all() and (0 <= self.cohort).all() and (self.cohort <= 1).all()
        return self


def tnlu(rng, low, high, size=None):
    """Truncated log-uniform draw (the TabPFN hyper-prior's workhorse)."""
    return np.exp(rng.uniform(np.log(low), np.log(high), size))


def reg2cls_binary(rng, target: np.ndarray, prevalence: float | None = None) -> np.ndarray:
    """Continuous target to a binary label: either by a prevalence quantile (credit range) or by a random boundary drawn
    from the target values (the TabPFN Reg2Cls recipe), class order random."""
    t = np.asarray(target, dtype=np.float64)
    if prevalence is None:
        b = rng.choice(t); y = (t > b).astype(np.int64)
    else:
        y = (t > np.quantile(t, 1 - prevalence)).astype(np.int64)
    if rng.random() < 0.5 and prevalence is None: y = 1 - y
    if y.min() == y.max():  # degenerate boundary: fall back to the median
        y = (t > np.median(t)).astype(np.int64)
    return y


def quantise(rng, col: np.ndarray, k: int) -> np.ndarray:
    """A continuous column into k categories with random (not equal-mass) boundaries, codes permuted (nominal) or kept (ordinal)."""
    finite = col[np.isfinite(col)]
    if len(finite) < k + 1: return col
    cuts = np.sort(rng.choice(finite, k - 1, replace=False)); codes = np.digitize(col, cuts).astype(np.float32)
    if rng.random() < 0.7: codes = rng.permutation(k)[codes.astype(int)].astype(np.float32)
    codes[~np.isfinite(col)] = np.nan
    return codes


def add_missing(rng, X: np.ndarray, p_table: float = 0.3) -> np.ndarray:
    """MCAR on a few columns, or value-dependent (MNAR) on one column, with probability p_table."""
    if rng.random() > p_table: return X
    X = X.copy(); n, d = X.shape; cols = rng.choice(d, max(1, int(rng.integers(1, max(2, d // 3 + 1)))), replace=False)
    if rng.random() < 0.6:
        rate = tnlu(rng, 0.01, 0.4)
        for c in cols: X[rng.random(n) < rate, c] = np.nan
    else:
        c = cols[0]; col = X[:, c]; thr = np.nanquantile(col, rng.uniform(0.5, 0.95))
        mask = (col > thr) if rng.random() < 0.5 else (col < np.nanquantile(col, rng.uniform(0.05, 0.5)))
        X[mask & (rng.random(n) < rng.uniform(0.5, 1.0)), c] = np.nan
    return X


def add_nuisance_columns(rng, X: np.ndarray, is_cat: np.ndarray, tags: np.ndarray, max_extra: int = 6):
    """Uninformative noise columns, collinear copies (same trait measured twice) and ratio / difference columns. Extra
    columns are tagged UNKNOWN except the collinear copy, which keeps its source's tag."""
    n, d = X.shape; cols, cats, tgs = [X], [is_cat], [tags]
    k_noise = int(rng.integers(0, 3)) if rng.random() < 0.5 else 0
    if k_noise: cols.append(rng.standard_normal((n, k_noise)).astype(np.float32)); cats.append(np.zeros(k_noise, bool)); tgs.append(np.zeros(k_noise, np.int64))
    if rng.random() < 0.3 and d >= 1:
        j = int(rng.integers(d)); noise = tnlu(rng, 0.05, 0.5)
        cols.append((X[:, [j]] + noise * np.nanstd(X[:, j]) * rng.standard_normal((n, 1))).astype(np.float32)); cats.append(np.zeros(1, bool)); tgs.append(np.array([tags[j]]))
    if rng.random() < 0.3 and d >= 2:
        num = [j for j in range(d) if not is_cat[j]]
        if len(num) >= 2:
            a, b = rng.choice(num, 2, replace=False); xa, xb = X[:, a], X[:, b]
            if rng.random() < 0.5:
                den = xb - np.nanmin(xb) + np.nanstd(xb) * 0.1 + 1e-6; r = (xa - np.nanmin(xa)) / den
            else:
                r = xa - xb
            cols.append(np.clip(r, -1e6, 1e6).astype(np.float32)[:, None]); cats.append(np.zeros(1, bool)); tgs.append(np.zeros(1, np.int64))
    X = np.concatenate(cols, 1)[:, :d + max_extra]; is_cat = np.concatenate(cats)[:X.shape[1]]; tags = np.concatenate(tgs)[:X.shape[1]]
    return X, is_cat, tags


def looks_categorical(col: np.ndarray) -> bool:
    """The same heuristic the real loaders use: at most 12 distinct integer values. Applied to synthetic columns too, so the
    categorical flag means the same thing on both sides (real books flag about 30% of columns, the
    simulators flagged 0 to 3%)."""
    v = col[np.isfinite(col)]
    return len(v) > 0 and len(np.unique(v)) <= 12 and np.all(np.abs(v - np.round(v)) < 1e-6)


def discretise(rng, X: np.ndarray, is_cat: np.ndarray, p_col: float = 0.6, p_floor: float = 0.45, p_skew: float = 0.3) -> np.ndarray:
    """Marginal realism for numeric columns (optional, off in the released model; independent draws per column):
    - skew (p_skew): a monotone exp transform, since SCM outputs are symmetric and real amounts are log-normal-like;
    - floor point mass (p_floor): values under a random quantile collapse to the minimum (zero-inflated counts, amounts,
      'months since'); real columns have a dominant value in 10%+ of rows about half the time;
    - quantile binning (p_col) into k levels, k log-uniform in [3, n/2]: few levels become integer codes (then flagged
      categorical), many levels keep their bin means. Real columns have about 5% distinct values, synthetic had 80 to 100%."""
    X = X.copy(); n = X.shape[0]
    for j in range(X.shape[1]):
        if is_cat[j]: continue
        col = X[:, j]; m = np.isfinite(col)
        if m.sum() < 8: continue
        c = col[m].astype(np.float64)
        if rng.random() < p_skew:
            z = (c - c.mean()) / (c.std() + 1e-9); c = np.exp(np.clip(z * rng.uniform(0.4, 1.5), -6, 6)) * rng.choice([-1.0, 1.0])
        if rng.random() < p_floor:
            floor = np.quantile(c, rng.uniform(0.1, 0.7)); c[c <= floor] = c.min()
        if rng.random() < p_col:
            k = int(round(tnlu(rng, 3, max(4, n / 2)))); edges = np.quantile(c, np.linspace(0, 1, k + 1)[1:-1]); codes = np.digitize(c, edges)
            if k <= 12: c = codes.astype(np.float64)
            else:
                means = np.array([c[codes == q].mean() if (codes == q).any() else np.nan for q in range(k)]); c = means[codes]
        X[m, j] = c.astype(np.float32)
    return X


def flip_labels(rng, y: np.ndarray, p_table: float = 0.5, rate_range=(0.02, 0.3)) -> np.ndarray:
    """Label noise: real default labels are partly unpredictable from any feature set (achievable AUC 0.55 to 0.85 on real
    books); synthetic labels were nearly deterministic. With probability p_table flip a log-uniform share of labels."""
    if rng.random() > p_table: return y
    y = y.copy(); pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m = int(min(round(tnlu(rng, *rate_range) * len(y) / 2), len(pos) - 1, len(neg) - 1))   # equal counts each way: prevalence preserved
    if m <= 0: return y
    y[rng.choice(pos, m, replace=False)] = 0; y[rng.choice(neg, m, replace=False)] = 1; return y


def sample_context_mask(rng, n: int, cohort: np.ndarray, frac_range=(0.3, 0.9), out_of_time: bool = False) -> np.ndarray:
    frac = rng.uniform(*frac_range); n_ctx = int(np.clip(round(frac * n), 2, n - 1))
    if out_of_time and cohort.max() > cohort.min():
        order = np.argsort(cohort + 1e-6 * rng.random(n)); ctx = np.zeros(n, bool); ctx[order[:n_ctx]] = True
        return ctx
    ctx = np.zeros(n, bool); ctx[rng.choice(n, n_ctx, replace=False)] = True; return ctx


def standardise(X: np.ndarray, context: np.ndarray, is_cat: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """z-score every numeric column on the context rows (NaN-aware), clip at +-clip sigma; categoricals are left as codes."""
    X = X.astype(np.float32, copy=True)
    for j in range(X.shape[1]):
        if is_cat[j]: continue
        c = X[context, j]; m = np.nanmean(c) if np.isfinite(c).any() else 0.0; s = np.nanstd(c) if np.isfinite(c).any() else 1.0
        s = s if s > 1e-8 else 1.0
        X[:, j] = np.clip((X[:, j] - m) / s, -clip, clip)
    return X


def finalize(rng, X, y, is_cat, cohort=None, meta=None, out_of_time_p: float = 0.5, nuisance: bool = True, missing: bool = True,
             ensure_both_classes: bool = True, tags=None, discretise_p: float = 0.0, label_noise_p: float = 0.0) -> Table:
    n, d = X.shape; X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.int64); is_cat = np.asarray(is_cat, bool)
    tags = np.zeros(d, np.int64) if tags is None else np.asarray(tags, np.int64)
    cohort = np.zeros(n, np.float32) if cohort is None else np.asarray(cohort, np.float32)
    if nuisance: X, is_cat, tags = add_nuisance_columns(rng, X, is_cat, tags)
    if missing: X = add_missing(rng, X)
    if discretise_p > 0: X = discretise(rng, X, is_cat, p_col=discretise_p)
    is_cat = is_cat | np.array([looks_categorical(X[:, j]) for j in range(X.shape[1])])     # same flag rule as the real loaders
    if label_noise_p > 0: y = flip_labels(rng, y, p_table=label_noise_p)
    oot = cohort.max() > cohort.min() and rng.random() < out_of_time_p
    for _ in range(20):
        ctx = sample_context_mask(rng, n, cohort, out_of_time=oot)
        if not ensure_both_classes or (y[ctx].min() != y[ctx].max()): break
    else:  # force one row of each class into the context
        ctx[np.flatnonzero(y == 1)[0]] = True; ctx[np.flatnonzero(y == 0)[0]] = True
    X = standardise(X, ctx, is_cat)
    rows = rng.permutation(n); cols = rng.permutation(X.shape[1])
    t = Table(X[rows][:, cols], y[rows], is_cat[cols], cohort[rows], ctx[rows], dict(meta or {}, out_of_time=bool(oot)), tags[cols])
    return t.check()
