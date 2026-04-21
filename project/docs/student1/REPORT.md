# Student 1 Report — Data, Experimental setup, Methods

This document covers the Student 1 responsibility area (phases 3.1 – 3.5)
of the tabular missing-data benchmark. Output tables and logs referenced
below live in `project/results/`.

---

## 1. Data

### 1.1 Selected datasets

Student 1 uses three binary-classification bankruptcy-prediction datasets
from the article's reference list. They span different sample counts and
feature dimensions on purpose, so that conclusions are not tied to a
single dataset size. All three are stored as standardised CSV files in
`project/data/processed/` and registered in `src/dataset_registry.py`.

| Dataset | File | Rows | Features | Classes | Native NaN |
|---|---|---|---|---|---|
| Taiwanese Bankruptcy Prediction (UCI id=572) | `taiwan_bankruptcy.csv` | ~6 819 | 95 | 2 | yes |
| Polish Companies Bankruptcy, 1-year horizon (UCI id=365) | `polish_1year.csv` | ~7 027 | 64 | 2 | yes |
| Slovak financial ratios (bankrupt + non-bankrupt, 2013 subset) | `slovak_manufacture_13.csv` | ~8 340 | 24 | 2 | yes |

Row counts above are the full standardised dataset; the exact figures
appear in `results/tables/dataset_overview.csv` after Phase 3.1.

### 1.2 Why these three

- All three are **real**, noisy financial tabular data with **natively
  missing values** — ideal for testing imputation and robustness.
- They cover a deliberate spread of scale:
  - Taiwan: medium rows, highest feature count (95);
  - Polish 1-year: medium rows, mid feature count (64);
  - Slovak: largest rows, lowest feature count (24).
- All are binary classification with a standardised `target` column,
  so a single pipeline handles them.
- They are **the same datasets** used by Student 2 (phases 4.1 – 4.7),
  which makes results directly comparable across the two students.

### 1.3 Train / test split, seed, metrics

- Split is produced once by `src/run_dataset_setup.py` and cached in
  `project/data/splits/<dataset>_{train,test}.csv`. All Student 1
  phases load the same cached split via
  `student1_common.load_precomputed_split`.
- `test_size = 0.2`, **stratified** on the target to preserve class
  proportions (see `config.TEST_SIZE`).
- `random_state = RANDOM_STATE = 42`, used everywhere: split,
  missingness injection, model seeds.
- Primary metrics (`config.METRICS`): `accuracy`, `f1`, `roc_auc`.
  Precision and recall are recorded additionally.
- Classification direction: `f1` is weighted (`average="weighted"`).
  `roc_auc` is binary; reported as `NaN` for non-binary cases.

---

## 2. Experimental setup

### 2.1 Missingness scenarios (Task 3.2)

Missingness is injected **only into the training split**; the test split
is kept intact so that test metrics remain comparable across scenarios.

- Mechanisms: `MCAR`, `MAR`, `MNAR` — all three are included so that
  Student 1 scenarios match Student 2 Phase 4.3+ one-to-one.
- Rates: `5%, 10%, 15%, 20%, 30%, 40%` — broader than the minimum
  required (10 % / 20 % / 30 % / 40 %) and identical to the Student 2
  rate grid.
- Implementation: `src/missingness.py`. Detailed description of MCAR /
  MAR / MNAR generation is in `docs/student1/missingness_mcar_mar.md`.
- Verification: `src/run_student1_3_2_verification.py` measures the
  **actual** final missing fraction for every scenario and stores it in
  `results/tables/student1_3_2_missingness_verification.{json,csv}`.
  Tolerance: `|actual - target| <= 0.01` (one percentage point).

### 2.2 Preprocessing / imputation (Task 3.3)

Three simple imputers are implemented in `student1_common.impute`:

- `mean` — per-column mean fitted on the (possibly NaN-containing)
  training set.
- `median` — per-column median fitted on the training set.
- `knn` — `sklearn.impute.KNNImputer(n_neighbors=5)` fitted on the
  training set and applied to both train and test.

All imputers are fit on training data only, to avoid leakage.
Artifacts: `results/tables/student1_3_3_imputation_results.{json,csv}`.

### 2.3 Models

#### Logistic Regression (Task 3.4, baseline)

- `sklearn.linear_model.LogisticRegression`
- `solver="lbfgs"`, `max_iter=1000`, `random_state=42`.
- Input is standardised with `StandardScaler` **after** imputation.
- Computation: CPU-only (sklearn).

#### Random Forest (Task 3.4, baseline)

- `sklearn.ensemble.RandomForestClassifier`
- `n_estimators=100`, `random_state=42`, `n_jobs=-1`.
- Input is standardised with `StandardScaler` **after** imputation
  (harmless for trees, kept for pipeline symmetry).
- Computation: CPU-only.

Artifacts for baselines: `results/tables/student1_3_4_baseline_results.{json,csv}`.

#### TabPFN (Task 3.5, foundation model)

- `tabpfn.TabPFNClassifier` (≥ 2.x; tested with 7.1.1).
- `ignore_pretraining_limits=True` so that all three datasets run despite
  possibly exceeding the original pretraining envelope.
- Preprocessing options (`TABPFN_PREPROCESSING`): `mean`, `median`, `knn`,
  `none`. The `"none"` path passes NaN directly to TabPFN
  (TabPFN v2+ has a built-in NaN handling preprocessing step).
- Device: auto-detected via `student1_common.detect_device`:
  CUDA if `torch.cuda.is_available()`, CPU otherwise.
- License: TabPFN 7.x requires a `TABPFN_TOKEN` environment variable
  (or cached accepted license). Unset → the run starts, logs a warning,
  and records a `TabPFNLicenseError` per scenario. Full note:
  `docs/student1/tabpfn_limitations.md`.
- Artifacts: `results/tables/student1_3_5_tabpfn_results.{json,csv}`.

### 2.4 Output format and Student 2 compatibility

Student 1 and Student 2 use the same base schema so that results can be
concatenated directly. CSV column order for Student 1 baselines (3.4) and
TabPFN (3.5) matches `results/tables/phase4_5_consolidated_results.csv`:

```
phase, dataset, model, model_type, preprocessing,
accuracy, f1, precision, recall, roc_auc,
training_time_seconds, missing_mechanism, missing_rate, ...
```

Additional Student-1-specific columns (`device`, `available`,
`native_nan_input`, `n_train_used`, `n_features_used`, `error`) are
appended at the end and are safe to ignore when concatenating with
Student 2 output.

`phase` values:

- `"3.3"` for imputation-only results,
- `"3.4"` for baseline models,
- `"3.5"` for TabPFN.

`model_type`:

- `"Classical"` for Logistic Regression / Random Forest (like Student 2
  Phase 4.5),
- `"Foundation"` for TabPFN.

### 2.5 Hardware / GPU usage

- GPU is auto-detected once at the start of each phase and logged
  (`"GPU detected: <name>"` or `"No GPU detected; using CPU"`).
- **Task 3.3** (imputers): CPU only; `StandardScaler` and `KNNImputer`
  are sklearn.
- **Task 3.4** (LogReg, RandomForest): CPU only by design (sklearn).
  The log still records that detection was performed.
- **Task 3.5** (TabPFN): uses CUDA when available, else CPU.
  The actual device chosen at runtime is stored per-row in the CSV
  under the `device` column, so one can later filter results by
  `device == "cuda"` / `device == "cpu"`.

---

## 3. Methods — short narrative

### 3.1 MCAR / MAR / MNAR generation

See `docs/student1/missingness_mcar_mar.md` for the full description.
Summary:

- **MCAR**: uniform sampling of observed numeric cells;
  no dependence on values.
- **MAR**: per column, masking probability depends on a different
  observed feature (control) — rows with control above the median
  have weight 0.8, others 0.2.
- **MNAR (synthetic)**: masking probability depends on the feature's
  own observed values, again with 0.8 / 0.2 weights above / below the
  median.
- Target rates refer to the approximate **final** missing fraction.
  Existing native NaN are preserved and counted.

### 3.2 Mean / Median / kNN imputation

- **Mean / median**: per-column statistic fitted on the training set and
  applied to both splits. Stateless, extremely fast.
- **kNN**: `KNNImputer(n_neighbors=5)` on numeric features. Slower but
  keeps feature correlation structure better than column-wise statistics.
- All three return NumPy-compatible matrices aligned on column names;
  imputation is followed by `StandardScaler` before baseline models.

### 3.3 Baseline models (Logistic Regression, Random Forest)

- Logistic Regression serves as a linear, well-calibrated baseline —
  sensitive to feature scale (hence the `StandardScaler`) and to
  imputation quality.
- Random Forest serves as a non-linear, tree-based baseline —
  scale-invariant, robust to outliers, but still requires a full
  (imputed) matrix because `sklearn` forests do not accept NaN natively.
- Together they bracket "simple linear" and "simple non-linear" so that
  the TabPFN comparison is informative on both ends.

### 3.4 TabPFN (foundation model)

- TabPFN is a pretrained transformer applied to tabular data via
  in-context learning: the full training set is passed to the model
  along with the test rows, and predictions are returned without
  parameter updates.
- For Student 1 we test TabPFN under the same scenarios as the
  baselines (native + MCAR / MAR / MNAR × rate grid) and additionally
  with `preprocessing == "none"` that feeds NaN directly to the model.
- Recorded per scenario: `accuracy`, `f1`, `precision`, `recall`,
  `roc_auc`, `training_time_seconds`, `device`, `native_nan_input`,
  `n_train_used`, `n_features_used`.
- Known limitations (size envelope, license token, compute time) are
  documented in `docs/student1/tabpfn_limitations.md`.

---

## 4. Artifacts summary (Student 1)

Code:

- `src/student1_common.py` — shared utilities (splits, impute,
  injection wrappers, TabPFN runner, device detection).
- `src/run_student1_3_2_verification.py` — Task 3.2 rate verification.
- `src/run_student1_3_3_experiments.py` — Task 3.3 imputers.
- `src/run_student1_3_4_experiments.py` — Task 3.4 baselines.
- `src/run_student1_3_5_experiments.py` — Task 3.5 TabPFN.
- `src/run_student1_pipeline.py` — orchestrator 3.2 → 3.3 → 3.4 → 3.5.

Tables:

- `results/tables/student1_3_2_missingness_verification.{json,csv}`
- `results/tables/student1_3_3_imputation_results.{json,csv}`
- `results/tables/student1_3_4_baseline_results.{json,csv}`
- `results/tables/student1_3_5_tabpfn_results.{json,csv}`

Docs:

- `docs/student1/REPORT.md` (this file)
- `docs/student1/missingness_mcar_mar.md`
- `docs/student1/tabpfn_limitations.md`

Logs: `results/logs/experiment_student1_*.log`.
