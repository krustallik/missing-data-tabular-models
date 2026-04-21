# Student 1 - Task 3.5: TabPFN limitations and operational notes

This note complements the results in
`results/tables/student1_3_5_tabpfn_results.{json,csv}` and the report in
`docs/student1/REPORT.md`. It collects the practical limitations of TabPFN
observed in this project (Phase 3.5 / Student 1 pipeline).

## TabPFN version used

- `tabpfn >= 2.x` (this project was tested against `tabpfn 7.1.1`).
- Installed via `pip install tabpfn` (see `project/requirements.txt`).

## Supported hardware and device selection

- TabPFN runs on **CPU** or **CUDA GPU** via PyTorch.
- The Student 1 pipeline auto-detects the device with
  `student1_common.detect_device()`:
  - returns `"cuda"` when `torch.cuda.is_available()` is `True`,
  - returns `"cpu"` otherwise.
- The selected device is passed to `TabPFNClassifier(device=...)` and is also
  stored in the output JSON/CSV (`device` column).
- GPU is used *only* by TabPFN (foundation model). Logistic Regression and
  Random Forest (Task 3.4) are CPU-only sklearn estimators by design.

## Dataset size / feature limits

TabPFN is a pretrained transformer. The original pretraining grid imposes
practical limits:

- **Training rows**: TabPFN's pretrained prior covers roughly up to
  ~10 000 training rows; the library emits a warning beyond that.
- **Features**: the pretrained grid covers up to ~100–500 features
  (depends on TabPFN version). Above this, TabPFN may fall back to
  internal feature subsampling.
- We pass `ignore_pretraining_limits=True` so that datasets exceeding these
  bounds still run. The effective `n_train_used` and `n_features_used` are
  recorded per-scenario in the JSON output.
- Student 1 datasets (see `docs/student1/REPORT.md`): Taiwan ≈ 6.8k rows,
  Polish 1-year ≈ 7.0k rows, Slovak manufacture ≈ 8.3k rows,
  all with < 100 features → within the typical envelope.

## Native handling of missing values

- TabPFN v2+ includes a built-in NaN-handling preprocessing step, so the
  model accepts `NaN` in the input arrays without an external imputer.
- Student 1 Task 3.5 exploits this: in addition to the three imputers
  (`mean`, `median`, `knn`), we add a `"none"` preprocessing path that
  passes the raw NaN-containing matrix directly to TabPFN. See
  `TABPFN_PREPROCESSING` in `src/run_student1_3_5_experiments.py`.
- Whether a row used native NaN input is stored in the output under
  `native_nan_input` (True / False).

## Licensing constraint (Prior Labs)

- TabPFN 7.x requires a one-time license acceptance to download the
  pretrained model weights. On a machine without a cached license this
  manifests as a `TabPFNLicenseError` on first `fit()`.
- Non-interactive workaround (used by this project and by Student 2 Phase
  4.4): set the environment variable `TABPFN_TOKEN` before running any
  TabPFN script, e.g.:

  ```powershell
  $env:TABPFN_TOKEN = "<your-api-key-from-ux.priorlabs.ai>"
  python src/run_student1_3_5_experiments.py
  ```

- The token is created at <https://ux.priorlabs.ai> → `Account`.
- `student1_common.test_tabpfn` now warns in the log when `TABPFN_TOKEN`
  is unset; the run will still start but `fit()` will fail with the
  license error if no cached weights are present.

## Computational cost

- TabPFN is inference-based ("in-context learning"). It does not train
  weights; the whole train set is passed to the transformer at inference.
- Typical wall time observed in Student 2 Phase 4.4 with the same splits:
  ~1400-1500 s on CPU for a ~5.5k-row training split and 95 features.
  On CUDA the wall time drops by roughly an order of magnitude depending
  on hardware (see `training_time_seconds` column in the JSON/CSV).
- Every (dataset × mechanism × rate × preprocessing) combination is a
  separate inference call. For Student 1 that is
  `3 datasets × (1 native + 3 × 6 missingness) × 4 preprocessing =
   3 × 19 × 4 = 228` calls.

## What the output JSON / CSV records

For every scenario:

- `device` — `cuda` or `cpu` actually used;
- `available` — whether `tabpfn` was importable at runtime;
- `native_nan_input` — True only for the `preprocessing == "none"` path;
- `n_train_used`, `n_features_used` — what TabPFN actually saw;
- `training_time_seconds` — end-to-end fit+predict time;
- `error` — any exception message (license / ImportError / internal).

## Summary of known limitations

1. License / token required for model weight download
   (`TABPFN_TOKEN` environment variable).
2. Pretraining envelope (~10k rows / ~100-500 features). We pass
   `ignore_pretraining_limits=True` to allow runs beyond it.
3. Training is inference — no parameter updates — so computation time
   scales with the training set size at every call, not amortised.
4. Native NaN handling is available in v2+ but is implementation-dependent;
   the "none" scenarios in our CSV show empirically whether it worked on
   each dataset.
