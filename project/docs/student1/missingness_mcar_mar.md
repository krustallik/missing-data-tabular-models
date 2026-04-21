# Student 1 - Task 3.2: MCAR and MAR generation

This document describes the missingness-generation logic used in Student 1
phases 3.2-3.5. The implementation lives in `project/src/missingness.py`.
MNAR is included alongside MCAR and MAR so that Student 1 scenarios match
Student 2 Phase 4.3+ one-to-one.

## Common rules

- Target column is excluded from masking.
- Only **numeric feature columns** are eligible for masking.
- Any **native** missing values already present in the data are preserved and
  counted towards the current missingness state.
- The requested rate refers to the approximate **final** missing fraction over
  numeric feature cells, not to the number of added cells.
- If the current missing fraction is already ≥ target, no extra masking is
  added (logged as a warning).
- Seed: `RANDOM_STATE = 42` (from `project/src/config.py`).
- Supported rates for Student 1: `5%, 10%, 15%, 20%, 30%, 40%` (aligned
  with Student 2 Phase 4.3).

## MCAR — Missing Completely At Random

`inject_mcar(X, target_final_missing_rate, random_state=42)`

1. Compute `total_cells = n_rows * n_numeric_cols` and
   `desired_missing = round(total_cells * target_final_missing_rate)`.
2. Compute the number of new missing cells needed:
   `missing_to_add = desired_missing - current_missing`.
3. Collect all **currently observed** (non-NaN) cells.
4. Sample `missing_to_add` cells uniformly at random, without replacement,
   using a `numpy.random.default_rng(random_state)` generator.
5. Set the chosen cells to `NaN`.

The mechanism has **no dependence** on any value (observed or unobserved),
which is the defining property of MCAR.

## MAR — Missing At Random

`inject_mar(X, target_final_missing_rate, random_state=42)`

For each numeric feature `f_i` (the "masked" feature), we choose another
numeric feature `f_j = f_{(i+1) mod k}` as its **control** (observable)
feature. Masking probability depends on the control, not on `f_i` itself.

1. For the control `f_j`, compute the median over observed rows.
2. Weight candidate rows by the control value:
   - `weight = 0.8` if `f_j(row) > median(f_j)`,
   - `weight = 0.2` otherwise.
3. Keep only rows where `f_i(row)` is still observed (NaN cannot be masked
   twice).
4. Normalise weights to a probability distribution per feature.
5. Across all (row, feature) candidates build a single weighted pool.
6. Sample `missing_to_add` positions without replacement using the pooled
   weights and set them to `NaN`.

Because the masking probability in `f_i` is a function of `f_j` (observed),
this satisfies MAR.

## MNAR — Missing Not At Random (synthetic)

`inject_mnar(X, target_final_missing_rate, random_state=42)`

Synthetic MNAR in this project uses the feature's **own observed values**
as the control: rows with values above the feature's own median get higher
masking probability (0.8 vs 0.2). The selected cells are then masked,
producing dependence between the probability of being missing and the value
that becomes missing. This is a standard synthetic MNAR construction used
across the missingness benchmarks.

## Verification (Task 3.2)

The script `project/src/run_student1_3_2_verification.py` checks, for every
dataset × mechanism × rate combination:

- the initial numeric missing fraction (train split),
- the final numeric missing fraction after injection,
- absolute deviation `final - target`,
- relative deviation `(final - target) / target`,
- whether the scenario is within tolerance (`|abs_dev| <= 0.01`).

Artifacts:

- `results/tables/student1_3_2_missingness_verification.csv`
- `results/tables/student1_3_2_missingness_verification.json`
- `results/logs/experiment_student1_3_2_verification_<timestamp>.log`

The JSON/CSV schema uses the Student 2 naming convention
(`dataset`, `missing_mechanism`, `missing_rate`).
