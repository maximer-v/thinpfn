"""The ten-book benchmark: context = n stratified rows of the train split (seeds), query = the validation split by default
(the test split only with --split test, which is the sealed, run-once protocol behind results/sealed_test.txt). Reports AUC
per book and n for ThinPFN and the baselines (logistic regression, XGBoost grid, TabPFN v2 if installed), with a paired
bootstrap CI of ThinPFN minus each baseline over the query rows (seeds averaged inside each resample).

    python benchmark/run_benchmark.py                          # validation split, all books, n = 200 / 500 / 1000
    python benchmark/run_benchmark.py --books us_bureau tvs --ns 200 --no-tabpfn
    python benchmark/run_benchmark.py --split test --sealed    # final evaluation: both flags required"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinpfn import ThinPFNClassifier, baselines  # noqa: E402
from thinpfn.data import books  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"


def boot(yv, A, Z, n_boot, rng):
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(yv), len(yv))
        if yv[i].min() == yv[i].max(): continue
        ds.append(np.mean([roc_auc_score(yv[i], a[i]) - roc_auc_score(yv[i], z[i]) for a, z in zip(A, Z)]))
    return float(np.quantile(ds, 0.025)), float(np.quantile(ds, 0.975))


def main(a):
    if a.split == "test" and not a.sealed: raise SystemExit("the test split is sealed: pass --sealed to confirm a final, run-once evaluation")
    base = {"logit": baselines.logistic, "xgb": baselines.xgboost}
    if not a.no_tabpfn:
        try: import tabpfn  # noqa: F401
        except ImportError: print("tabpfn not installed: skipping the TabPFN v2 baseline (pip install tabpfn)", flush=True)
        else: base["tabpfn"] = baselines.tabpfn_v2
    clf = ThinPFNClassifier(weights=a.weights, n_perm=a.perm, preprocess=a.preprocess); rows = []; t0 = time.time()
    for book in a.books:
        Xtr, ytr = books.load(book, "train"); Xq, yq = books.load(book, a.split, sealed_ok=(a.split == "test"))
        if len(yq) > a.query_cap: i = np.random.default_rng(0).choice(len(yq), a.query_cap, replace=False); Xq, yq = Xq[i], yq[i]
        for n in a.ns:
            if n > len(ytr): continue
            S = {"thinpfn": [], **{k: [] for k in base}}
            for s in range(a.seeds):
                Xc, yc = books.subsample_context(Xtr, ytr, n, s)
                S["thinpfn"].append(clf.fit(Xc, yc).decision_function(Xq))
                for k, fn in base.items():
                    try: S[k].append(fn(Xc, yc, Xq, seed=s))
                    except Exception as e:
                        if not a.allow_missing: raise
                        print(f"{book}|{n}|{s}|{k}: {type(e).__name__} {e}", flush=True); S[k].append(np.full(len(yq), np.nan))
            rec = dict(book=book, n=n, n_query=len(yq), prevalence=round(float(yq.mean()), 3))
            for k, v in S.items():   # a baseline with any failed seed is reported as missing for the whole cell (never a mean over fewer seeds)
                ok = [sc for sc in v if not np.isnan(sc).any()]; rec[k] = float(np.mean([roc_auc_score(yq, sc) for sc in ok])) if len(ok) == a.seeds else float("nan")
                rec[f"constant_{k}"] = int(sum(len(np.unique(np.round(sc, 6))) == 1 for sc in ok))   # seeds whose scorer collapsed to one value (AUC 0.5)
            for k in base:
                if not np.isnan(rec[k]): rec[f"d_{k}"] = rec["thinpfn"] - rec[k]; rec[f"lo_{k}"], rec[f"hi_{k}"] = boot(yq, S["thinpfn"], S[k], a.boot, np.random.default_rng(0))
            rows.append(rec); print(json.dumps(rec), flush=True)
    df = pd.DataFrame(rows)
    L = [f"ThinPFN benchmark on the {a.split} split ({a.seeds} context seeds, query <= {a.query_cap} rows, {a.boot} paired bootstrap resamples), {time.strftime('%Y-%m-%d')}", ""]
    L.append(f"{'book':14s} {'n':>5s} {'thinpfn':>8s} " + " ".join(f"{k:>7s}  delta [95% CI]        " for k in base))
    for r in rows:
        line = f"{r['book']:14s} {r['n']:5d} {r['thinpfn']:8.3f} "
        for k in base:
            if f"d_{k}" in r: sig = "*" if (r[f"lo_{k}"] > 0 or r[f"hi_{k}"] < 0) else " "; line += f"{r[k]:7.3f}  {r[f'd_{k}']:+.3f} [{r[f'lo_{k}']:+.3f},{r[f'hi_{k}']:+.3f}]{sig} "
            else: line += f"{r.get(k, float('nan')):7.3f}  {'n/a':24s} "
        L.append(line)
    L.append("")
    for n in a.ns:
        g = df[df.n == n]
        if len(g): L.append(f"n={n}: mean over {len(g)} books: thinpfn {g.thinpfn.mean():.4f}  " + "  ".join(f"{k} {g[k].mean():.4f}" for k in base))
    for k in base:
        if f"d_{k}" in df:
            L.append(f"thinpfn vs {k}: ahead {(df[f'd_{k}'] > 0).sum()}/{len(df)} cells, significantly ahead {(df[f'lo_{k}'] > 0).sum()}, significantly behind {(df[f'hi_{k}'] < 0).sum()}" + (f"; missing cells {df[k].isna().sum()}" if df[k].isna().any() else ""))
            deg = df[df[f"constant_{k}"] > 0]
            if len(deg): L.append(f"   {k} collapsed to a constant score in {int(deg[f'constant_{k}'].sum())} seed fits: " + ", ".join(f"{r.book}@{r.n}" for r in deg.itertuples()))
    txt = "\n".join(L); print(txt); RESULTS.mkdir(exist_ok=True); out = RESULTS / (a.out or f"benchmark_{a.split}.txt"); out.write_text(txt + "\n"); df.to_csv(out.with_suffix(".csv"), index=False)
    print(f"written {out} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--books", nargs="+", default=books.BOOK_NAMES); ap.add_argument("--ns", nargs="+", type=int, default=[200, 500, 1000]); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split", default="validation", choices=["validation", "test"]); ap.add_argument("--sealed", action="store_true"); ap.add_argument("--query-cap", type=int, default=4000); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--weights", default="thinpfn_v1"); ap.add_argument("--perm", type=int, default=8); ap.add_argument("--preprocess", default="qnorm"); ap.add_argument("--no-tabpfn", action="store_true"); ap.add_argument("--out", default="")
    ap.add_argument("--allow-missing", action="store_true", help="report a failed baseline as missing instead of stopping (default: stop)")
    main(ap.parse_args())
