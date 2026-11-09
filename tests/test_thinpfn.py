"""python -m pytest -q tests  (CPU is enough; the prior and the model run on CPU for small tables)."""
import numpy as np
import pytest
import torch

from thinpfn import PFNConfig, ThinPFN, ThinPFNClassifier, load_pretrained
from thinpfn.priors import credit_ledger, mixture, mlp_scm, tree_scm
from thinpfn.priors.contract import Table


def _check_table(t: Table):
    assert t.X.ndim == 2 and t.X.shape[0] == t.n and t.X.shape[1] == t.d and t.X.dtype == np.float32
    assert set(np.unique(t.y)) <= {0, 1} and t.context.dtype == bool and t.context.sum() >= 2 and (~t.context).sum() >= 1
    assert t.is_categorical.shape == (t.d,) and t.cohort.shape == (t.n,)
    assert np.isfinite(t.X[t.context]).any()


@pytest.mark.parametrize("sampler", [mlp_scm.sample, tree_scm.sample, credit_ledger.sample])
def test_priors_produce_valid_tables(sampler):
    rng = np.random.default_rng(0)
    for _ in range(5): _check_table(sampler(rng, n=256))


def test_mixture_and_collate():
    pm = mixture.PriorMixture("default", n_range=(256, 384), d_max=64, seed=1); tables = pm.batch(4)
    b = mixture.collate(tables, d_max=64)
    assert b["X"].shape[0] == 4 and b["X"].shape[2] == 64 and b["feature_mask"].any(1).all()
    assert {t.meta["mixture_component"] for t in tables} <= set(mixture.DEFAULT_WEIGHTS)


def test_small_model_forward_and_predict_cpu():
    torch.manual_seed(0); model = ThinPFN(PFNConfig(col_blocks=0, row_layers=0, d_icl=64, icl_layers=2, icl_heads=4, n_think=4)).eval()
    rng = np.random.default_rng(0); Xc = rng.standard_normal((100, 5)).astype(np.float32); yc = (Xc[:, 0] > 0).astype(int); Xq = rng.standard_normal((30, 5)).astype(np.float32)
    s = model.predict(Xc, yc, Xq, device=torch.device("cpu"), is_cat=False)
    assert s.shape == (30,) and np.isfinite(s).all()
    s2 = model.predict(Xc, yc, Xq, device=torch.device("cpu"), is_cat=False, preprocess="qnorm", n_perm=3)
    assert s2.shape == (30,) and np.isfinite(s2).all()


def test_query_rows_do_not_see_each_other():
    """Scores of a query row must not depend on which other query rows are in the batch."""
    torch.manual_seed(0); model = ThinPFN(PFNConfig(col_blocks=0, row_layers=0, d_icl=64, icl_layers=2, icl_heads=4, n_think=4)).eval()
    rng = np.random.default_rng(1); Xc = rng.standard_normal((80, 4)).astype(np.float32); yc = (Xc[:, 1] > 0).astype(int); Xq = rng.standard_normal((20, 4)).astype(np.float32)
    a = model.predict(Xc, yc, Xq, device=torch.device("cpu"), is_cat=False); b = model.predict(Xc, yc, Xq[:5], device=torch.device("cpu"), is_cat=False)
    assert np.allclose(a[:5], b, atol=1e-4)


def test_released_weights_load_and_learn_in_context():
    model = load_pretrained(device="cpu"); assert model.n_params() > 25_000_000
    rng = np.random.default_rng(2); X = rng.standard_normal((700, 6)).astype(np.float32); y = ((X[:, 0] + 0.5 * X[:, 1] ** 2 - 0.3 * X[:, 2]) > 0.6).astype(int)
    clf = ThinPFNClassifier(device="cpu", n_perm=2).fit(X[:400], y[:400]); p = clf.predict_proba(X[400:])[:, 1]
    from sklearn.metrics import roc_auc_score
    assert p.shape == (300,) and roc_auc_score(y[400:], p) > 0.85
    assert clf.predict(X[400:]).shape == (300,)


def test_classifier_accepts_pandas_with_object_columns():
    import pandas as pd
    rng = np.random.default_rng(3); n = 300
    df = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.choice(["x", "y", "z"], n), "c": rng.integers(0, 5, n).astype(float)}); df.loc[:10, "a"] = np.nan
    y = (df["a"].fillna(0) + (df["b"] == "x") > 0.5).astype(int).to_numpy()
    clf = ThinPFNClassifier(device="cpu", n_perm=1).fit(df[:200], y[:200]); p = clf.predict_proba(df[200:])
    assert p.shape == (100, 2) and np.allclose(p.sum(1), 1)


def test_sealed_test_split_guard():
    from thinpfn.data import books
    with pytest.raises(PermissionError): books.load("uci_full", "test")


def test_lower_level_predict_with_heuristic_flags_and_string_device():
    model = load_pretrained(device="cpu"); rng = np.random.default_rng(4)
    Xc = np.c_[rng.standard_normal((120, 3)), rng.integers(0, 4, 120)].astype(np.float32); yc = (Xc[:, 0] + Xc[:, 3] > 1).astype(int); Xq = Xc[:20]
    s = model.predict(Xc, yc, Xq, device="cpu")           # default is_cat=None flags the integer column
    assert s.shape == (20,) and np.isfinite(s).all()
    assert model.predict(Xc, yc, Xq[:0], device="cpu").shape == (0,)


def test_classifier_sklearn_conventions_and_masks():
    from sklearn.base import is_classifier, clone
    rng = np.random.default_rng(5); X = np.c_[rng.standard_normal((300, 3)), rng.integers(0, 3, 300)].astype(np.float32); y = (X[:, 0] > 0).astype(int)
    clf = ThinPFNClassifier(device="cpu", n_perm=1, categorical=[False, False, False, True])
    assert is_classifier(clf) and clone(clf).get_params()["n_perm"] == 1
    clf.fit(X[:200], y[:200]); assert clf.score(X[200:], y[200:]) > 0.8
    auto = ThinPFNClassifier(device="cpu", n_perm=1, categorical="auto").fit(X[:200], y[:200]); assert auto.predict_proba(X[200:]).shape == (100, 2)
    with pytest.raises(ValueError): ThinPFNClassifier(device="cpu", categorical=[True]).fit(X, y)
    with pytest.raises(ValueError): ThinPFNClassifier(device="cpu").fit(X, y[:10])
    with pytest.raises(ValueError): clf.set_params(nonsense=1)
    clf.set_params(weights="does-not-exist")
    with pytest.raises(FileNotFoundError): clf.model


def test_classifier_context_is_copied_and_capped_exactly():
    rng = np.random.default_rng(6); X = rng.standard_normal((500, 4)).astype(np.float32); y = (X[:, 1] > 0.5).astype(int)
    clf = ThinPFNClassifier(device="cpu", n_perm=0, max_context=100).fit(X, y)
    assert clf.X_.shape == (100, 4) and set(np.unique(clf.y_)) == {0, 1}
    X[0, 0] = 99.0; assert clf.X_[0, 0] != 99.0 or clf.X_.base is not X
    with pytest.raises(ValueError): ThinPFNClassifier(device="cpu", max_context=1).fit(X, y)


def test_classifier_string_dtype_and_consistent_encoding():
    import pandas as pd
    rng = np.random.default_rng(8); n = 240
    df = pd.DataFrame({"a": rng.standard_normal(n), "b": pd.array(rng.choice(["x", "y", "z"], n), dtype="string"), "c": rng.choice([True, False], n)})
    y = ((df["a"] > 0) | (df["b"] == "x")).astype(int).to_numpy()
    clf = ThinPFNClassifier(device="cpu", n_perm=1).fit(df[:160], y[:160])
    p = clf.predict_proba(df[160:])[:, 1]; assert p.shape == (80,)
    unseen = df[160:].copy(); unseen.loc[unseen.index[:5], "b"] = "w"      # unseen category becomes NaN, no error
    assert np.isfinite(clf.predict_proba(unseen)).all()
    with pytest.raises(ValueError): clf.predict_proba(df[160:].drop(columns=["c"]))


def test_calibration_metrics():
    from thinpfn import metrics
    rng = np.random.default_rng(9); p = rng.uniform(0.02, 0.6, 20000); y = (rng.uniform(size=p.size) < p).astype(int); z = np.log(p / (1 - p))
    good = metrics.calibration(y, z); shifted = metrics.calibration(y, z + 1.5)
    assert good["ece"] < 0.02 and shifted["ece"] > 0.15 and good["brier"] < shifted["brier"]
    assert abs(good["auc"] - shifted["auc"]) < 1e-12 and abs(good["ks"] - shifted["ks"]) < 1e-12      # rank metrics ignore the shift
    assert 0 < good["brier_skill"] < 1 and abs(good["mean_pred"] - good["prevalence"]) < 0.02
    assert abs(metrics.calibration(y, np.full(y.size, np.log(y.mean() / (1 - y.mean()))))["brier_skill"]) < 1e-9   # base-rate forecast has zero skill
    mp, obs, cnt = metrics.reliability_bins(y, p, 10); assert len(mp) == 10 and cnt.sum() == y.size and np.all(np.diff(mp) > 0)


def test_classifier_works_inside_sklearn_calibration_and_cv():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    rng = np.random.default_rng(10); X = rng.standard_normal((240, 4)).astype(np.float32); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    base = ThinPFNClassifier(device="cpu", n_perm=0)
    assert cross_val_score(base, X, y, cv=2, scoring="roc_auc").mean() > 0.85
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=2).fit(X[:180], y[:180]); p = cal.predict_proba(X[180:])
    assert p.shape == (60, 2) and np.allclose(p.sum(1), 1)
