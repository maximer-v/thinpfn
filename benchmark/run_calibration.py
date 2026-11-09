"""Probability quality on the validation split: are the scores usable as probabilities without a calibration step?

Same contexts as the benchmark (n stratified rows of the train split, seeds 0..2) and the same query rows (validation
split, at most 4,000). For ThinPFN and each baseline it reports, per book and n and averaged over seeds: AUC, KS, Brier
score, Brier skill against the constant base-rate forecast, expected calibration error (10 equal-count bins) and the mean
predicted probability next to the observed prevalence. No recalibration is applied to any model. The sealed test split
is not read: it was used once, for the AUC claims, and this study was added afterwards.

    python benchmark/run_calibration.py
    python benchmark/run_calibration.py --books us_bureau tvs --ns 500 --no-tabpfn

Writes results/calibration.csv (one row per book, n, model), results/calibration_bins.csv (reliability curve per model
and n, rows pooled over books and seeds) and results/calibration.txt."""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinpfn import ThinPFNClassifier, baselines, metrics  # noqa: E402
from thinpfn.data import books  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
KEYS = ("auc", "ks", "brier", "brier_skill", "ece", "log_loss", "mean_pred", "prevalence")


def main(a):
    scorers = {"logit": baselines.logistic, "xgb": baselines.xgboost}
    if not a.no_tabpfn:
        try: import tabpfn  # noqa: F401
        except ImportError: print("tabpfn not installed: skipping the TabPFN v2 baseline (pip install tabpfn)", flush=True)
        else: scorers["tabpfn"] = baselines.tabpfn_v2
    clf = ThinPFNClassifier(weights=a.weights); rows = []; pool = {}; t0 = time.time()
    for book in a.books:
        Xtr, ytr = books.load(book, "train"); Xq, yq = books.load(book, "validation")
        if len(yq) > a.query_cap: i = np.random.default_rng(0).choice(len(yq), a.query_cap, replace=False); Xq, yq = Xq[i], yq[i]
        for n in a.ns:
            if n > len(ytr): continue
            per = {k: [] for k in ("thinpfn", *scorers)}
            for s in range(a.seeds):
                Xc, yc = books.subsample_context(Xtr, ytr, n, s)
                S = {"thinpfn": clf.fit(Xc, yc).decision_function(Xq), **{k: fn(Xc, yc, Xq, seed=s) for k, fn in scorers.items()}}
                for k, sc in S.items():
                    per[k].append(metrics.calibration(yq, sc)); pool.setdefault((k, n), []).append((yq, metrics.sigmoid(sc)))
            for k, ms in per.items():
                rows.append(dict(book=book, n=n, model=k, **{key: float(np.mean([m[key] for m in ms])) for key in KEYS}))
            print(f"{book:14s} n={n:5d} " + "  ".join(f"{k} ece {np.mean([m['ece'] for m in per[k]]):.3f} bss {np.mean([m['brier_skill'] for m in per[k]]):+.3f}" for k in per), flush=True)
    df = pd.DataFrame(rows); bins = []
    for (k, n), parts in pool.items():
        y = np.concatenate([p[0] for p in parts]); p = np.concatenate([p[1] for p in parts]); mp, obs, cnt = metrics.reliability_bins(y, p, a.bins)
        bins += [dict(model=k, n=n, bin=j, mean_pred=float(mp[j]), observed=float(obs[j]), count=int(cnt[j])) for j in range(len(mp))]
    L = [f"ThinPFN probability quality on the validation split ({a.seeds} context seeds, query <= {a.query_cap} rows, no recalibration), {time.strftime('%Y-%m-%d')}", "",
         "mean over books of each metric, by context size and model (lower is better for ece, brier, log_loss; higher for the rest)", ""]
    L.append(f"{'n':>5s} {'model':8s} {'auc':>7s} {'ks':>7s} {'brier':>7s} {'skill':>7s} {'ece':>7s} {'logloss':>8s} {'mean_p':>7s} {'prev':>7s}")
    for n in a.ns:
        for k in df.model.unique():
            g = df[(df.n == n) & (df.model == k)]
            if len(g): L.append(f"{n:5d} {k:8s} {g.auc.mean():7.4f} {g.ks.mean():7.4f} {g.brier.mean():7.4f} {g.brier_skill.mean():+7.4f} {g.ece.mean():7.4f} {g.log_loss.mean():8.4f} {g.mean_pred.mean():7.4f} {g.prevalence.mean():7.4f}")
        L.append("")
    L.append("books where the model is worse than the base-rate forecast (Brier skill < 0), by n:")
    for n in a.ns:
        L.append(f"  n={n}: " + "; ".join(f"{k} {int((df[(df.n == n) & (df.model == k)].brier_skill < 0).sum())}/{int(((df.n == n) & (df.model == k)).sum())}" for k in df.model.unique()))
    txt = "\n".join(L); print(txt); RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration.txt").write_text(txt + "\n"); df.to_csv(RESULTS / "calibration.csv", index=False); pd.DataFrame(bins).to_csv(RESULTS / "calibration_bins.csv", index=False)
    print(f"written {RESULTS / 'calibration.txt'} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--books", nargs="+", default=books.BOOK_NAMES); ap.add_argument("--ns", nargs="+", type=int, default=[200, 500, 1000]); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--query-cap", type=int, default=4000); ap.add_argument("--bins", type=int, default=10); ap.add_argument("--weights", default="thinpfn_v1"); ap.add_argument("--no-tabpfn", action="store_true")
    main(ap.parse_args())
