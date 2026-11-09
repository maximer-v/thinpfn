"""The credit ledger simulator: a prior component that generates portfolios the generic SCM priors never produce.

Each table is a portfolio. A hyper-prior draws the product, the horizon and label definition, the base hazard, the
prevalence target, the observed feature blocks, documentation quality and drift magnitude. Each borrower has latent
traits that map monotonically (with random nonlinearity and log-normal / point-mass marginals) to application and
bureau features, and drives a monthly Markov chain of delinquency states from which bills and payments follow with
accounting identities. The label is default within the horizon after the observation month, with the hazard tied to
the chain; censoring is applied per the portfolio's label definition. Prevalence is set by bisection on the hazard
intercept. Drift is a macro regime path over the cohort index that moves the hazard and the payment behaviour.

Delinquency codes follow the UCI convention: -2 no consumption, -1 paid in full, 0 revolving (minimum paid), 1..8
months late. The hyper-parameter ranges below are the released ones (DEFAULT_TARGETS); a dict with the same keys can be passed
to sample() to recalibrate them from aggregate statistics of a portfolio."""
from __future__ import annotations

import numpy as np

from .contract import finalize, tnlu
from ..schema import simulator_tag_ids

DEFAULT_TARGETS = {
    "prevalence": [0.02, 0.35],
    "months_observed": [3, 12],
    "horizon_months": [1, 12],
    "oracle_auc": [0.62, 0.92],
    "thin_file_share": [0.0, 0.4],
    "slip_risk": [0.6, 1.6],
    "blocks": {"application": 0.9, "bureau": 0.6, "ledger": 0.85, "ratios": 0.5, "demographics": 0.6, "date": 0.3},
}
STATES = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])      # index 0..10
S_NOCONS, S_FULL, S_REV, S_LATE1 = 0, 1, 2, 3


def load_targets() -> dict:
    return dict(DEFAULT_TARGETS)


def transition_matrix(risk: np.ndarray, macro: np.ndarray, h: dict) -> np.ndarray:
    """Per-borrower-month transition matrices [n, 11, 11] over the delinquency states. Higher risk and worse macro
    raise the chance of slipping one state later and lower the chance of curing."""
    n = len(risk); logit_slip = h["slip_base"] + h["slip_risk"] * risk + h["slip_macro"] * macro
    p_slip = 1 / (1 + np.exp(-logit_slip))                     # probability of getting one month later / entering arrears
    p_cure = 1 / (1 + np.exp(-(h["cure_base"] - h["cure_risk"] * risk - h["cure_macro"] * macro)))
    p_full = 1 / (1 + np.exp(-(h["full_base"] - h["full_risk"] * risk)))
    p_nocons = h["nocons"]
    T = np.zeros((n, 11, 11))
    # from no consumption / paid in full / revolving (current states)
    for s in (S_NOCONS, S_FULL, S_REV):
        T[:, s, S_LATE1] = p_slip
        rest = 1 - p_slip
        T[:, s, S_NOCONS] = rest * p_nocons; T[:, s, S_FULL] = rest * (1 - p_nocons) * p_full; T[:, s, S_REV] = rest * (1 - p_nocons) * (1 - p_full)
    # from late k: cure to revolving (partial) or full, stay, or slip to k+1; 90+ days late is near-absorbing (write-off, rare cure)
    for k in range(1, 9):
        s = S_REV + k; pc = p_cure if k < 3 else p_cure * 0.1; stay = (1 - pc) * (1 - p_slip); slip = (1 - pc) * p_slip
        T[:, s, s] = stay; T[:, s, min(s + 1, 10)] += slip
        T[:, s, S_REV] = pc * (1 - p_full * 0.5); T[:, s, S_FULL] = pc * p_full * 0.5
    T /= T.sum(2, keepdims=True)
    return T


def simulate_chain(rng, T0_fn, n: int, months: int, risk, macro_path):
    """Run the chain for `months` months; macro_path [months] (per month macro value, same for the portfolio cohort, but
    each borrower's cohort offset picks a window). Returns states [n, months]."""
    states = np.full((n, months), S_REV, dtype=int)
    start = rng.choice([S_NOCONS, S_FULL, S_REV, S_LATE1], n, p=[0.15, 0.35, 0.42, 0.08]); states[:, 0] = start
    for m in range(1, months):
        T = T0_fn(risk, macro_path[:, m]); cur = states[:, m - 1]
        u = rng.random(n); cum = np.cumsum(T[np.arange(n), cur], 1); states[:, m] = (u[:, None] > cum).sum(1).clip(0, 10)
    return states


def sample_hyper(rng, targets):
    d_app = int(rng.integers(2, 7)); d_bur = int(rng.integers(2, 6)); months = int(rng.integers(*targets["months_observed"])) if rng.random() < 0.85 else 6
    return dict(
        product=rng.choice(["card", "instalment", "p2p", "vehicle"]), months=months, horizon=int(rng.integers(targets["horizon_months"][0], targets["horizon_months"][1] + 1)),
        prevalence=float(rng.uniform(*targets["prevalence"])), thin_file=float(rng.uniform(*targets["thin_file_share"])) if rng.random() < 0.5 else 0.0,
        d_app=d_app, d_bur=d_bur, n_traits=int(rng.integers(2, 5)),
        slip_base=float(rng.uniform(-4.0, -2.0)), slip_risk=float(rng.uniform(*targets["slip_risk"])), slip_macro=float(rng.uniform(0.0, 0.8)),
        cure_base=float(rng.uniform(-0.5, 1.0)), cure_risk=float(rng.uniform(0.3, 1.2)), cure_macro=float(rng.uniform(0.0, 0.6)),
        full_base=float(rng.uniform(-0.5, 1.0)), full_risk=float(rng.uniform(0.3, 1.0)), nocons=float(rng.uniform(0.05, 0.25)),
        drift=float(tnlu(rng, 0.02, 1.5)) if rng.random() < 0.6 else 0.0, trait_noise=float(tnlu(rng, 0.2, 1.5)),
        feature_noise=float(tnlu(rng, 0.1, 1.0)), recent_first=bool(rng.random() < 0.5), label_def="ever_30",
        blocks={k: bool(rng.random() < p) for k, p in targets["blocks"].items()}, limit_log_mean=float(rng.uniform(9, 12.5)), demographics_effect=float(rng.uniform(0, 0.15)),
    )


def monotone_map(rng, z, noise):
    """Random monotone nonlinearity of a latent plus noise, then a marginal transform (log-normal, point mass at zero, ordinal)."""
    kind = rng.choice(["linear", "exp", "sat", "step"]); s = rng.choice([-1, 1])
    v = s * z + noise * rng.standard_normal(len(z))
    if kind == "exp": v = np.exp(np.clip(v, -4, 4) * rng.uniform(0.3, 1.0))
    elif kind == "sat": v = np.tanh(v * rng.uniform(0.5, 2))
    elif kind == "step": v = np.digitize(v, np.quantile(v, np.sort(rng.random(int(rng.integers(2, 6)))))).astype(float)
    marg = rng.random()
    if marg < 0.3:   # log-normal money-like
        v = np.exp(rng.uniform(5, 11) + (v - v.mean()) / (v.std() + 1e-6) * rng.uniform(0.3, 1.2))
    if rng.random() < 0.25:  # point mass at zero (no bureau record, no payment)
        v = np.where(rng.random(len(v)) < rng.uniform(0.05, 0.4), 0.0, v)
    return v


def sample(rng, n=None, d=None, prevalence=None, finalize_table=True, targets=None, fin=None):
    targets = targets or load_targets(); h = sample_hyper(rng, targets)
    # a label needs at least as many horizon months as its days-past-due threshold (90+ needs 3 months late)
    feasible = [d for d, m in (("ever_30", 1), ("ever_60", 2), ("ever_90", 3), ("end_state", 3)) if h["horizon"] >= m]; h["label_def"] = str(rng.choice(feasible))
    if prevalence is not None: h["prevalence"] = prevalence
    n = n or int(rng.integers(256, 1025)); M = h["months"] + h["horizon"]
    # latent traits and risk
    traits = rng.standard_normal((n, h["n_traits"])); w = rng.standard_normal(h["n_traits"]); w /= np.linalg.norm(w)
    risk = traits @ w + h["trait_noise"] * rng.standard_normal(n); risk = (risk - risk.mean()) / risk.std()
    # cohort and macro path: cohort in [0,1]; macro worsens along cohort by drift, with a smooth random path
    cohort = rng.random(n) if h["drift"] > 0 else np.zeros(n)
    k = rng.standard_normal(6) * 0.3; months_axis = (np.arange(M) / M)[None, :] * 0.2
    phase = cohort[:, None] + months_axis           # each borrower's months occupy a window starting at its cohort
    macro = h["drift"] * (phase - 0.5) + 0.3 * sum(k[i] * np.sin((i + 1) * np.pi * phase) for i in range(6))
    T_fn = lambda r, mac: transition_matrix(r, mac, h)
    # bisection on the slip intercept so that prevalence hits the target
    lo, hi = -9.0, 4.0; base = h["slip_base"]
    for _ in range(12):
        h["slip_base"] = 0.5 * (lo + hi); st = simulate_chain(rng, T_fn, n, M, risk, macro); y = label_from_states(st, h)
        if y.mean() < h["prevalence"]: lo = h["slip_base"]
        else: hi = h["slip_base"]
    h["slip_base"] = 0.5 * (lo + hi); states = simulate_chain(rng, T_fn, n, M, risk, macro); y = label_from_states(states, h)
    if y.min() == y.max():  # fall back: label by hazard quantile
        y = (risk > np.quantile(risk, 1 - h["prevalence"])).astype(np.int64)
    obs = states[:, :h["months"]]                  # observed months; the horizon part only made the label
    # bills and payments from the observed chain with accounting identities
    limit = np.exp(h["limit_log_mean"] + 0.6 * rng.standard_normal(n) - 0.3 * risk)
    spend_rate = np.clip(0.2 + 0.15 * risk + 0.1 * rng.standard_normal(n), 0.02, 0.9)
    bill = np.zeros((n, h["months"])); pay = np.zeros((n, h["months"])); b = limit * np.clip(rng.uniform(0, 0.6, n) + 0.1 * risk, 0, 1)
    for m in range(h["months"]):
        s = obs[:, m]; spend = np.where(s == S_NOCONS, 0.0, spend_rate * limit * rng.uniform(0.5, 1.5, n))
        frac = np.where(s == S_FULL, 1.0, np.where(s == S_REV, rng.uniform(0.03, 0.5, n), np.where(s >= S_LATE1, rng.uniform(0, 0.05, n), 0.0)))
        p = frac * b; bill[:, m] = b; pay[:, m] = p; b = np.maximum(b - p + spend, 0.0)
    cols, names, is_cat = [], [], []
    bl = h["blocks"]
    if bl["ledger"]:
        order = range(h["months"]) if h["recent_first"] else reversed(range(h["months"]))
        order = list(order)[::-1] if h["recent_first"] else list(order)  # recent_first: last observed month first
        seq = obs[:, order]; cols += [STATES[seq[:, i]].astype(float) for i in range(h["months"])]; names += [f"pay_{i}" for i in range(h["months"])]; is_cat += [False] * h["months"]
        if rng.random() < 0.8:
            cols += [bill[:, order][:, i] for i in range(h["months"])]; names += [f"bill_{i}" for i in range(h["months"])]; is_cat += [False] * h["months"]
            cols += [pay[:, order][:, i] for i in range(h["months"])]; names += [f"payamt_{i}" for i in range(h["months"])]; is_cat += [False] * h["months"]
    if bl["application"]:
        cols.append(limit); names.append("limit"); is_cat.append(False)
        for j in range(h["d_app"]):
            t = traits[:, rng.integers(h["n_traits"])]; cols.append(monotone_map(rng, t, h["feature_noise"])); names.append(f"app_{j}"); is_cat.append(False)
    if bl["bureau"]:
        for j in range(h["d_bur"]):
            cols.append(monotone_map(rng, risk, h["feature_noise"] * 1.5)); names.append(f"bur_{j}"); is_cat.append(False)
    if bl["ratios"] and bl["ledger"]:
        last_bill = bill[:, -1]; last_pay = pay[:, -1]
        cols.append(last_bill / (limit + 1)); names.append("utilisation"); is_cat.append(False)
        cols.append(last_pay / (last_bill + 1)); names.append("pay_ratio"); is_cat.append(False)
    if bl["demographics"]:
        for j in range(int(rng.integers(1, 4))):
            kcat = int(rng.integers(2, 6)); c = rng.integers(0, kcat, n).astype(float)
            cols.append(c); names.append(f"demo_{j}"); is_cat.append(True)
        age = np.clip(40 + 12 * rng.standard_normal(n) - 3 * risk * h["demographics_effect"] * 10, 18, 80); cols.append(age); names.append("age"); is_cat.append(False)
    if bl["date"] and h["drift"] > 0:
        cols.append(cohort + 0.02 * rng.standard_normal(n)); names.append("application_date"); is_cat.append(False)
    if not cols:  # at least the ledger
        cols = [STATES[obs[:, i]].astype(float) for i in range(h["months"])]; names = [f"pay_{i}" for i in range(h["months"])]; is_cat = [False] * h["months"]
    X = np.stack(cols, 1).astype(np.float32); is_cat = np.array(is_cat, bool)
    # thin files: ledger block missing for a share of borrowers (informative: thin files skew to the young / low tenure)
    if h["thin_file"] > 0 and bl["ledger"]:
        thin = rng.random(n) < h["thin_file"]; ledger_cols = [i for i, nm in enumerate(names) if nm.startswith(("pay_", "bill_", "payamt_", "utilisation", "pay_ratio"))]
        X[np.ix_(thin, ledger_cols)] = np.nan
    if d is not None and X.shape[1] > d:
        keep = np.sort(rng.choice(X.shape[1], d, replace=False)); X = X[:, keep]; is_cat = is_cat[keep]; names = [names[i] for i in keep]
    meta = dict(component="credit_ledger", product=str(h["product"]), months=h["months"], horizon=h["horizon"], label_def=str(h["label_def"]), drift=h["drift"],
                prevalence_target=h["prevalence"], prevalence=float(y.mean()), names=names, oracle_auc=float(_auc(risk, y)))
    tags = np.array(simulator_tag_ids(names), np.int64)
    if rng.random() < 0.3: tags = np.zeros_like(tags)      # sometimes the lender gives no field types: the model must work without them
    if not finalize_table: return X, y.astype(np.int64), is_cat, cohort.astype(np.float32), meta
    return finalize(rng, X, y.astype(np.int64), is_cat, cohort.astype(np.float32), meta, nuisance=rng.random() < 0.5, tags=tags, **(fin or {}))


def label_from_states(states: np.ndarray, h: dict) -> np.ndarray:
    fut = states[:, h["months"]:]                 # the horizon months
    thr = {"ever_90": S_REV + 3, "ever_60": S_REV + 2, "ever_30": S_REV + 1, "end_state": S_REV + 3}[h["label_def"]]
    if h["label_def"] == "end_state": return (fut[:, -1] >= thr).astype(np.int64)
    return (fut.max(1) >= thr).astype(np.int64)


def _auc(score, y):
    y = np.asarray(y); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = np.argsort(np.argsort(score)) + 1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
