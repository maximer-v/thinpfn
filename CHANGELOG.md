# Changelog

All notable changes to ThinPFN. The model weights are versioned separately from the package: every release below ships
the same `thinpfn_v1` weights.

## 1.1.0 (2026-11-09)

Same model, same weights, same sealed-test numbers. This release adds evidence about the probabilities, makes the model
installable in one line, and puts the tests under continuous integration.

### Added

- **Probability study** (`benchmark/run_calibration.py`): KS, Brier score, Brier skill against the base-rate forecast,
  expected calibration error and a pooled reliability curve for ThinPFN, logistic regression, XGBoost and TabPFN v2,
  on the validation split of the ten books with no recalibration. Results in `benchmark/results/calibration.*`, a new
  README section and a new subsection in the paper. The sealed test split is not read again.
- **Latency table** (`benchmark/latency.py`): measured time of one scoring call on CPU and GPU for five context and
  query sizes, with and without the column-order ensemble. Replaces the single figure quoted in 1.0.0.
- `thinpfn.metrics`: `ks`, `ece`, `reliability_bins` and `calibration`.
- Install from GitHub without cloning: `pip install "git+https://github.com/maximer-v/thinpfn@v1.1.0"` (the weights ship
  inside the package).
- Continuous integration: the test suite on Python 3.10 and 3.12, plus a non-editable install that loads the packaged
  weights and scores a table.
- Tests for the new metrics and for use inside `CalibratedClassifierCV` and `cross_val_score`.
- Two figures (`calibration`, `latency`) in `benchmark/make_figures.py`.

### Changed

- README: probability section, latency table, a limitation on probabilities following the context default rate.
- Paper: probability-quality subsection with a table and a figure, latency paragraph, updated abstract, limitations
  and reproducibility sections.

### Unchanged

- The weights (`thinpfn_v1`), the inference recipe and every number of the sealed test.

## 1.0.0 (2026-11-09)

First public release: the synthetic prior (structural causal generators and a credit ledger simulator), the 25.6M
network and its trainer, the released weights, the scikit-learn style classifier, the ten-book benchmark with split
manifests and the sealed-test results against logistic regression, XGBoost and TabPFN v2, the walkthrough notebook and
the paper.
