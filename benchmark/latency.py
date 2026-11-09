"""Wall-clock latency of ThinPFNClassifier on synthetic numeric tables (20 features), median of repeated calls after one
warm-up. One call is one predict_proba over all query rows, context included; fit only stores rows and is not timed.

    python benchmark/latency.py                 # CPU (8 threads) and, if present, the GPU
    python benchmark/latency.py --devices cpu --threads 4

Writes results/latency.csv."""
from __future__ import annotations

import argparse, platform, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinpfn import ThinPFNClassifier  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
SHAPES = [(200, 1), (200, 1000), (1000, 1), (1000, 2000), (4000, 4000)]   # (context rows, query rows)


def main(a):
    torch.set_num_threads(a.threads); rng = np.random.default_rng(0); rows = []
    devices = a.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    for dev in devices:
        name = torch.cuda.get_device_name(0) if dev == "cuda" else f"{platform.machine()} CPU, {a.threads} threads"
        for n_ctx, n_q in SHAPES:
            X = rng.standard_normal((n_ctx + n_q, a.features)).astype(np.float32); y = (X[:n_ctx, 0] + rng.standard_normal(n_ctx) > 0).astype(int)
            for n_perm in (0, 8):
                clf = ThinPFNClassifier(device=dev, n_perm=n_perm).fit(X[:n_ctx], y); clf.predict_proba(X[n_ctx:]); ts = []
                for _ in range(a.repeats):
                    if dev == "cuda": torch.cuda.synchronize()
                    t = time.perf_counter(); clf.predict_proba(X[n_ctx:])
                    if dev == "cuda": torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t)
                rows.append(dict(device=dev, hardware=name, context=n_ctx, query=n_q, n_perm=n_perm, seconds=round(float(np.median(ts)), 4))); print(rows[-1], flush=True)
    RESULTS.mkdir(exist_ok=True); pd.DataFrame(rows).to_csv(RESULTS / "latency.csv", index=False); print(f"written {RESULTS / 'latency.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--devices", nargs="+", default=None); ap.add_argument("--threads", type=int, default=8); ap.add_argument("--features", type=int, default=20); ap.add_argument("--repeats", type=int, default=3)
    main(ap.parse_args())
