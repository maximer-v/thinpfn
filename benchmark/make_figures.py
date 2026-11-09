"""Figures for the README (docs/assets/*.png, titled) and the paper (paper/figures/*.pdf, untitled) from benchmark/results.
    python benchmark/make_figures.py"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RES = HERE / "results"; ASSETS = ROOT / "docs" / "assets"; PAPER = ROOT / "paper" / "figures"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
COLORS = {"thinpfn": "#2a78d6", "tabpfn": "#eb6834", "xgb": "#1baf7a", "logit": "#eda100"}
NAMES = {"thinpfn": "ThinPFN", "tabpfn": "TabPFN v2", "xgb": "XGBoost", "logit": "Logistic"}
BOOK = {"uci_full": "Taiwan cards (full)", "tw_cards": "Taiwan cards (4k train)", "us_bureau": "US bureau (GMSC)", "in_vehicle": "India vehicle (L&T)", "us_homeequity": "US home equity (HMEQ)",
        "us_p2p": "US P2P 2007-15", "lending_club": "LendingClub 2007-18", "ppdai": "China P2P (PPDai)", "prosper": "Prosper", "tvs": "India cross-sell (TVS)"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def spread(values: dict, gap: float) -> dict:
    """Vertical label positions that keep at least `gap` between neighbours (sorted ascending, pushed upward)."""
    order = sorted(values, key=values.get); pos = {}
    for i, k in enumerate(order): pos[k] = values[k] if i == 0 else max(values[k], pos[order[i - 1]] + gap)
    return pos


def save(fig, name, title=None):
    ASSETS.mkdir(parents=True, exist_ok=True); PAPER.mkdir(parents=True, exist_ok=True)
    if title: fig.suptitle(title, x=0.01, y=1.03, ha="left", fontsize=13, color=INK, fontweight="semibold")
    fig.savefig(ASSETS / f"{name}.png", dpi=180, bbox_inches="tight")
    if title: fig.suptitle("")
    fig.savefig(PAPER / f"{name}.pdf", bbox_inches="tight"); plt.close(fig)


def fig_mean_auc(df):
    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ns = sorted(df.n.unique())
    ypos = spread({k: df[df.n == ns[-1]][k].mean() for k in COLORS}, 0.004)
    for k in ["thinpfn", "tabpfn", "xgb", "logit"]:
        m = [df[df.n == n][k].mean() for n in ns]; ax.plot(ns, m, color=COLORS[k], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5)
        ax.annotate(f"{NAMES[k]}  {m[-1]:.3f}", (ns[-1], ypos[k]), xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9.5)
    ax.set_ylim(top=max(ypos.values()) + 0.005)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.set_xlabel("labelled rows in context (n)"); ax.set_ylabel("mean AUC over ten books (sealed test)")
    ax.set_xlim(170, 1700); ax.minorticks_off(); ax.grid(axis="x", visible=False)
    save(fig, "mean_auc", "Mean test AUC over ten public credit books")


def fig_forest(df, baseline="tabpfn", n=200):
    g = df[df.n == n].copy(); g["label"] = g.book.map(BOOK); g = g.sort_values(f"d_{baseline}")
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); y = np.arange(len(g))
    for i, r in enumerate(g.itertuples()):
        lo, hi, d = getattr(r, f"lo_{baseline}"), getattr(r, f"hi_{baseline}"), getattr(r, f"d_{baseline}")
        sig = lo > 0 or hi < 0; col = COLORS["thinpfn"] if d > 0 else COLORS[baseline]
        ax.plot([lo, hi], [i, i], color=col, lw=2 if sig else 1.2, alpha=1 if sig else 0.55, solid_capstyle="round")
        ax.plot(d, i, "o", color=col, ms=7 if sig else 5.5, mec=SURFACE, mew=1.2)
        ax.annotate(f"{d:+.3f}", (max(hi, 0.0) + 0.004, i), va="center", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_yticks(y); ax.set_yticklabels(g.label); ax.set_xlabel(f"AUC difference, ThinPFN minus {NAMES[baseline]} (n = {n}; 95% paired bootstrap)")
    ax.grid(axis="y", visible=False); ax.set_xlim(min(g[f"lo_{baseline}"].min(), -0.02) - 0.01, max(g[f"hi_{baseline}"].max(), 0.02) + 0.03)
    save(fig, f"forest_{baseline}_n{n}", f"Per-book difference against {NAMES[baseline]} at n = {n}")


def fig_grid(df):
    """Status grid: for each baseline, book x n cells: significantly ahead / ahead / behind / significantly behind."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.4), sharey=True); books = list(BOOK); ns = sorted(df.n.unique())
    palette = {2: "#2a78d6", 1: "#a9c8ee", -1: "#f5c1ad", -2: "#eb6834"}
    for ax, k in zip(axes, ["tabpfn", "xgb", "logit"]):
        for i, b in enumerate(books):
            for j, n in enumerate(ns):
                r = df[(df.book == b) & (df.n == n)].iloc[0]; d, lo, hi = r[f"d_{k}"], r[f"lo_{k}"], r[f"hi_{k}"]
                s = 2 if lo > 0 else -2 if hi < 0 else (1 if d > 0 else -1)
                ax.add_patch(plt.Rectangle((j + 0.04, i + 0.04), 0.92, 0.92, color=palette[s], lw=0)); ax.text(j + 0.5, i + 0.5, f"{d:+.2f}", ha="center", va="center", fontsize=8, color=INK if abs(s) == 1 else SURFACE)
        ax.set_xlim(0, len(ns)); ax.set_ylim(len(books), 0); ax.set_xticks(np.arange(len(ns)) + 0.5); ax.set_xticklabels([f"n={n}" for n in ns]); ax.set_yticks(np.arange(len(books)) + 0.5); ax.set_yticklabels([BOOK[b] for b in books])
        ax.set_title(f"vs {NAMES[k]}", fontsize=10.5, color=INK, loc="left"); ax.grid(False); ax.tick_params(length=0)
        for sp in ax.spines.values(): sp.set_visible(False)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=palette[2], label="ahead, CI above 0"), Patch(color=palette[1], label="ahead"), Patch(color=palette[-1], label="behind"), Patch(color=palette[-2], label="behind, CI below 0")],
               loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(); save(fig, "cells_grid", "Every cell of the sealed test: AUC difference of ThinPFN against each baseline")


def fig_training():
    recs = [json.loads(l) for l in open(RES / "training_log.jsonl")]; L = pd.DataFrame([r for r in recs if "held_loss" in r]); D = [r for r in recs if "dev" in r and r["step"] != "lawa"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax = axes[0]; ax.plot(L.step, L.held_loss, color=COLORS["thinpfn"], lw=1.8); ax.axhline(L.held_prior.iloc[0], color=INK2, lw=0.9, ls="--"); ax.annotate("class-prior loss", (L.step.max(), L.held_prior.iloc[0]), xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    ax.set_xlabel("optimiser step (64 tables each)"); ax.set_ylabel("held-out synthetic loss"); ax.set_ylim(0.3, 0.75)
    ax = axes[1]; steps = [r["step"] for r in D]; mean = [r["dev"]["mean"] for r in D]; ax.plot(steps, mean, color=COLORS["thinpfn"], lw=1.8, marker="o", ms=5, mec=SURFACE)
    ends = {"mean": mean[-1]}
    for b, col in [("us_homeequity@1000", COLORS["xgb"]), ("in_vehicle@1000", COLORS["logit"])]:
        v = [r["dev"][b] for r in D]; ax.plot(steps, v, color=col, lw=1.4, marker="o", ms=4, mec=SURFACE); ends[b] = v[-1]
    ypos = spread(ends, 0.02)
    for b, lab in [("mean", "six-book mean"), ("us_homeequity@1000", "us_homeequity (n=1000)"), ("in_vehicle@1000", "in_vehicle (n=1000)")]:
        ax.annotate(lab, (steps[-1], ypos[b]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("optimiser step"); ax.set_ylabel("development AUC (validation splits)"); ax.set_xlim(0, 21000)
    fig.tight_layout(); save(fig, "training", "Pre-training on 1.02M synthetic tables: the real-book AUC follows the synthetic loss")


def fig_inference():
    a = pd.read_csv(RES / "inference_tweaks.csv"); b = pd.read_csv(RES / "inference_tweaks_combo.csv")
    steps = [("z-score, single pass", a.base.mean()), ("+ quantile-normal inputs", a["cohort0+qnorm"].mean()), ("+ 8 column permutations", b["cohort0+qnorm+perm8"].mean())]
    fig, ax = plt.subplots(figsize=(7.2, 2.9)); y = np.arange(len(steps)); base_v = steps[0][1]
    for i, (lab, v) in enumerate(steps):   # dot plot (the axis does not start at zero, so no bars)
        if i: ax.plot([base_v, v], [i, i], color="#a9c8ee", lw=2, solid_capstyle="round")
        ax.plot(v, i, "o", color=COLORS["thinpfn"] if i == 2 else "#6ea3e3", ms=9, mec=SURFACE, mew=1.5); ax.text(v + 0.002, i, f"{v:.3f}" + (f"  ({v - base_v:+.3f})" if i else ""), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([s[0] for s in steps]); ax.invert_yaxis(); ax.set_xlim(0.68, 0.725); ax.set_xlabel("mean validation AUC over ten books and three context sizes (same 3.3M weights)"); ax.grid(axis="y", visible=False)
    save(fig, "inference", "Two inference-time changes, no retraining: +0.022 AUC")


def fig_scale():
    d = pd.read_csv(RES / "development_small_vs_v1.csv"); ns = sorted(d.n.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); series = [("v1", "3.3M, 160k tables", "#a9c8ee"), ("deep", "25.6M, 1.02M tables (ThinPFN)", COLORS["thinpfn"]), ("tabpfn", "TabPFN v2", COLORS["tabpfn"])]
    ypos = spread({k: d[d.n == ns[-1]][k].mean() for k, _, _ in series}, 0.004)
    for k, lab, col in series:
        m = [d[d.n == n][k].mean() for n in ns]; ax.plot(ns, m, color=col, lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5); ax.annotate(f"{lab}  {m[-1]:.3f}", (ns[-1], ypos[k]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_ylim(top=max(ypos.values()) + 0.005)
    ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns]); ax.minorticks_off(); ax.set_xlim(170, 2600); ax.set_xlabel("labelled rows in context (n)"); ax.set_ylabel("mean validation AUC, ten books"); ax.grid(axis="x", visible=False)
    save(fig, "scale", "Same prior, same inference: depth and compute move the mean (validation splits)")


if __name__ == "__main__":
    df = pd.read_csv(RES / "sealed_test.csv")
    fig_mean_auc(df); fig_forest(df, "tabpfn", 200); fig_forest(df, "xgb", 200); fig_grid(df); fig_training(); fig_inference(); fig_scale()
    print("figures written to", ASSETS, "and", PAPER)
