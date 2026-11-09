"""MLP structural-causal-model prior, after TabPFN v1 (tabpfn/priors/mlp.py, Apache-2.0) and TabICL (BSD-3), re-implemented.

A table is drawn as: a random layered network of latent nodes (causes at the input), random sparse weights, one
activation per table drawn from a menu that includes smooth, piecewise and periodic functions, Gaussian noise at every
node, per-layer standardisation with a random affine rescale (TabICL). Features are a random subset of the hidden
nodes (optionally a contiguous block, TabPFN's in_clique), the target is one hidden node not among the features. The
continuous target becomes a binary label by Reg2Cls or a prevalence quantile. Some feature columns are quantised to
categoricals. The drift wrapper can perturb a sparse subset of the weights along the cohort index."""
from __future__ import annotations

import numpy as np

from .contract import finalize, quantise, reg2cls_binary, tnlu

ACTIVATIONS = {
    "tanh": np.tanh, "relu": lambda x: np.maximum(x, 0), "leaky": lambda x: np.where(x > 0, x, 0.1 * x), "elu": lambda x: np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)), "sin": np.sin, "abs": np.abs, "square": lambda x: np.sign(x) * np.minimum(x * x, 50), "identity": lambda x: x,
    "step": lambda x: (x > 0).astype(np.float64), "softplus": lambda x: np.log1p(np.exp(np.minimum(x, 30))), "cube": lambda x: np.clip(x, -5, 5) ** 3 / 10,
    "gelu": lambda x: 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x ** 3))), "sawtooth": lambda x: x - np.floor(x) - 0.5,
}


def sample_hyper(rng, n_features: int):
    return dict(
        layers=int(np.clip(round(tnlu(rng, 1, 6)), 1, 6)), hidden=int(np.clip(round(tnlu(rng, 8, 130)), max(8, n_features + 2), 130)),
        causes=int(np.clip(round(tnlu(rng, 1, 12)), 1, 12)), init_std=float(tnlu(rng, 0.05, 5)), noise_std=float(tnlu(rng, 1e-3, 0.3)),
        sparsity=float(0.9 * rng.beta(rng.uniform(0.1, 5), rng.uniform(0.1, 5))), activation=rng.choice(list(ACTIVATIONS)),
        in_clique=bool(rng.random() < 0.3), gp_input=bool(rng.random() < 0.3), rescale=bool(rng.random() < 0.7),
    )


class MLPSCM:
    """Holds the sampled network so the drift wrapper can perturb its weights; __call__ generates rows for a cohort value."""

    def __init__(self, rng, n_features: int, h: dict):
        self.rng, self.h, self.d = rng, h, n_features
        self.W = []
        for l in range(h["layers"]):
            fan_in = h["causes"] if l == 0 else h["hidden"]
            W = rng.standard_normal((fan_in, h["hidden"])) * h["init_std"] / np.sqrt(fan_in)
            W *= (rng.random(W.shape) > h["sparsity"]); self.W.append(W)
        self.b = [rng.standard_normal(h["hidden"]) * 0.1 for _ in range(h["layers"])]
        self.rescale = [(np.exp(2 * rng.uniform(-0.5, 0.5)), rng.uniform(-1, 1)) for _ in range(h["layers"])]
        nodes = rng.permutation(h["hidden"])
        if h["in_clique"]:
            start = int(rng.integers(0, h["hidden"] - n_features)); self.feature_nodes = np.arange(start, start + n_features); rest = np.setdiff1d(nodes, self.feature_nodes)
        else:
            self.feature_nodes = nodes[:n_features]; rest = nodes[n_features:]
        self.target_node = int(rest[0]) if len(rest) else int(nodes[-1])
        self.feature_layer = [int(rng.integers(0, h["layers"])) for _ in range(n_features)]
        self.target_layer = h["layers"] - 1
        self.act = ACTIVATIONS[h["activation"]]

    def forward(self, n: int, weight_delta=None):
        rng, h = self.rng, self.h
        if h["gp_input"]:
            t = np.sort(rng.random(n)); ls = tnlu(rng, 0.05, 1.0)
            K = np.exp(-0.5 * ((t[:, None] - t[None, :]) / ls) ** 2) + 1e-6 * np.eye(n); L = np.linalg.cholesky(K)
            x = (L @ rng.standard_normal((n, h["causes"])))
        else:
            x = rng.standard_normal((n, h["causes"]))
        acts = []
        for l in range(h["layers"]):
            W = self.W[l] + (weight_delta[l] if weight_delta is not None else 0)
            z = x @ W + self.b[l] + h["noise_std"] * rng.standard_normal((n, h["hidden"]))
            z = self.act(z)
            z = (z - z.mean(0)) / (z.std(0) + 1e-6)
            if h["rescale"]: a, b = self.rescale[l]; z = a * (z + b)
            acts.append(z); x = z
        X = np.stack([acts[self.feature_layer[j]][:, self.feature_nodes[j]] for j in range(self.d)], 1)
        target = acts[self.target_layer][:, self.target_node] + 0.1 * rng.standard_normal(n)
        return X, target


def sample(rng, n: int | None = None, d: int | None = None, prevalence: float | None = None, drift=None, finalize_table: bool = True, fin=None):
    """One MLP-SCM table. drift: optional callable(scm, n, rng) -> (X, target, cohort) implemented in drift.py."""
    n = n or int(rng.integers(256, 1025)); d = d or int(np.clip(round(tnlu(rng, 2, 64)), 2, 64))
    h = sample_hyper(rng, d); scm = MLPSCM(rng, d, h)
    if drift is not None:
        X, target, cohort = drift(scm, n, rng)
    else:
        X, target = scm.forward(n); cohort = None
    if prevalence is None and rng.random() < 0.5: prevalence = float(rng.uniform(0.02, 0.5))
    y = reg2cls_binary(rng, target, prevalence)
    is_cat = np.zeros(d, bool)
    if rng.random() < 0.3:
        for j in rng.choice(d, int(rng.integers(1, max(2, d // 2 + 1))), replace=False):
            k = int(max(2, round(rng.gamma(1, 10)))); k = min(k, 12); X[:, j] = quantise(rng, X[:, j], k); is_cat[j] = True
    meta = dict(component="mlp_scm", **{k: (v if not isinstance(v, np.generic) else v.item()) for k, v in h.items()}, prevalence_target=prevalence)
    if not finalize_table: return X.astype(np.float32), y, is_cat, cohort, meta
    return finalize(rng, X, y, is_cat, cohort, meta, **(fin or {}))
