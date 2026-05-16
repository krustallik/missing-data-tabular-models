"""Standalone randomized-search pipeline for the 8 classical models.

Independent from ``run_experiments.py``. Samples a wide hyperparameter space
for every tunable classical model from :data:`config.CLASSICAL_MODELS`, then
evaluates the same sampled candidates across the three bankruptcy datasets
with three different CV seeds.

Design notes
~~~~~~~~~~~~
- **Metric**: primary scoring is ``average_precision`` (PR-AUC) — it is
  threshold-independent and is the recommended summary for highly
  imbalanced binary classification (positive rate 1-4 %). We also record
  ``balanced_accuracy`` for cross-reference.
- **Search**: 150 random parameter candidates per model by default. The same
  candidates are reused for every dataset so the universal ranking compares
  like with like.
- **Seeds**: ``[42, 123, 7]``. For each (dataset, model, seed) we build a
  fresh ``StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)``.
- **Imbalance**: ``class_weight='balanced'`` (or equivalent) is fixed where
  the estimator supports it. ``GradientBoosting`` and
  ``MLP`` have no class-weight knob; we rely on PR-AUC (ranking metric,
  not affected by threshold) instead of accuracy.
- **Universal best**: parameters that consistently rank well across all
  three datasets. We compute the within-dataset rank of every parameter
  combo (averaged across seeds), then average ranks across datasets. The
  combo with the best mean rank wins; we also report mean and min PR-AUC
  across datasets as diagnostics for stability.
- **Foundation models**: TabPFN and TabICL are pretrained and not tuned
  here. The pipeline targets the 8 tunable classical models.

Outputs (under ``results/tables/random_search/``):
- ``random_search_full_results.csv``     - every (dataset, model, seed, params).
- ``random_search_top_per_dataset.csv``  - best params per (dataset, model)
                                           averaged across seeds.
- ``random_search_universal.csv``        - best universal params per model
                                           across the three datasets.

Usage::

    python src/random_search_pipeline.py
    python src/random_search_pipeline.py --models random_forest xgboost
    python src/random_search_pipeline.py --datasets taiwan_bankruptcy
    python src/random_search_pipeline.py --seeds 42 7
    python src/random_search_pipeline.py --n-iter 150
    python src/random_search_pipeline.py --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, loguniform, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterSampler, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    CLASSICAL_MODELS,
    DATASETS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    ensure_output_dirs,
)
from data_utils import (
    coerce_features,
    load_dataset_from_csv,
    load_precomputed_split,
    sanitize_feature_columns,
    setup_logging,
)


# ── Configuration ────────────────────────────────────────────────────────────

SEEDS: List[int] = [42, 123, 7]
CV_FOLDS: int = 5
DEFAULT_N_ITER: int = 150

# PR-AUC is threshold-independent and is the recommended ranking metric for
# very imbalanced binary problems (1-4% positives in our datasets).
PRIMARY_METRIC = "pr_auc"
SCORING = {
    "pr_auc": "average_precision",
    "balanced_accuracy": "balanced_accuracy",
}

# All datasets we want searched (independent of what config has uncommented
# in DATASETS - random search runs on the full set).
GRID_DATASETS: Dict[str, Path] = {
    "polish_1year": PROCESSED_DIR / "polish_1year.csv",
    "slovak_manufacture_13": PROCESSED_DIR / "slovak_manufacture_13.csv",
    "taiwan_bankruptcy": PROCESSED_DIR / "taiwan_bankruptcy.csv",
}

OUT_DIR = TABLES_DIR / "random_search"
OUT_FULL = OUT_DIR / "random_search_full_results.csv"
OUT_TOP_PER_DATASET = OUT_DIR / "random_search_top_per_dataset.csv"
OUT_UNIVERSAL = OUT_DIR / "random_search_universal.csv"


# ── Random-search spaces ─────────────────────────────────────────────────────
#
# Spaces are intentionally wider than the old grid. We sample exactly
# ``n_iter`` candidates per model, then evaluate those fixed candidates on
# every dataset. Steps prefixed by ``clf__`` operate on the
# classifier inside a Pipeline.

def _build_estimator_and_space(
    model_key: str,
    y_train: pd.Series,
    seed: int,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    """Return ``(estimator, param_space)`` or ``None`` if dependency missing."""

    # Every classical model that does not accept NaN natively is wrapped
    # in a Pipeline starting with ``SimpleImputer(strategy="median")`` so
    # that the search runs on the raw splits (which may contain a few
    # natural NaN cells, especially in Slovak Manufacture 13).
    imputer = SimpleImputer(strategy="median")

    if model_key == "logistic_regression":
        est = Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs", max_iter=3000, class_weight="balanced",
            )),
        ])
        space = {
            "clf__C": loguniform(1e-4, 1e3),
            "clf__class_weight": ["balanced", None],
        }
        return est, space

    if model_key == "random_forest":
        est = Pipeline([
            ("imputer", imputer),
            ("clf", RandomForestClassifier(
                n_jobs=1, class_weight="balanced", random_state=seed,
            )),
        ])
        space = {
            "clf__n_estimators": randint(200, 1201),
            "clf__max_depth": [None, 4, 6, 8, 10, 12, 16, 20, 30, 40],
            "clf__min_samples_split": randint(2, 31),
            "clf__min_samples_leaf": randint(1, 16),
            "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0],
            "clf__bootstrap": [True, False],
            "clf__criterion": ["gini", "entropy", "log_loss"],
            "clf__class_weight": ["balanced", "balanced_subsample"],
        }
        return est, space

    if model_key == "gradient_boosting":
        # GradientBoosting has no class_weight knob; PR-AUC (ranking) is
        # the right metric here so we tune the boosting hyperparameters only.
        est = Pipeline([
            ("imputer", imputer),
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ])
        space = {
            "clf__n_estimators": randint(100, 1001),
            "clf__learning_rate": loguniform(0.01, 0.3),
            "clf__max_depth": randint(2, 8),
            "clf__min_samples_split": randint(2, 51),
            "clf__min_samples_leaf": randint(1, 31),
            "clf__subsample": uniform(0.5, 0.5),
            "clf__max_features": [None, "sqrt", "log2", 0.5, 0.8],
        }
        return est, space

    if model_key == "svm":
        # SVC is the slowest model, but randomized search keeps it bounded.
        est = Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=seed)),
        ])
        space = {
            "clf__C": loguniform(1e-3, 1e3),
            "clf__gamma": loguniform(1e-4, 1.0),
            "clf__kernel": ["rbf"],
            "clf__shrinking": [True, False],
            "clf__tol": loguniform(1e-5, 1e-2),
        }
        return est, space

    if model_key == "mlp":
        est = Pipeline([
            ("imputer", imputer),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                max_iter=700, early_stopping=True, random_state=seed,
            )),
        ])
        space = {
            "clf__hidden_layer_sizes": [
                (50,), (100,), (200,), (100, 50), (200, 100), (256, 128),
                (100, 50, 25),
            ],
            "clf__alpha": loguniform(1e-6, 1e-1),
            "clf__learning_rate_init": loguniform(1e-4, 1e-2),
            "clf__activation": ["relu", "tanh"],
            "clf__solver": ["adam"],
            "clf__batch_size": [32, 64, 128, 256, "auto"],
            "clf__learning_rate": ["constant", "adaptive"],
            "clf__beta_1": uniform(0.8, 0.19),
            "clf__beta_2": uniform(0.9, 0.099),
        }
        return est, space

    if model_key == "xgboost":
        try:
            import xgboost as xgb
        except ImportError:
            return None
        # scale_pos_weight is dataset-specific; fix it from the training labels.
        y_arr = np.asarray(y_train)
        n_pos = int((y_arr == 1).sum())
        n_neg = int((y_arr != 1).sum())
        spw = float(n_neg) / float(n_pos) if n_pos else 1.0
        est = xgb.XGBClassifier(
            n_jobs=1, eval_metric="logloss", verbosity=0, tree_method="hist",
            scale_pos_weight=spw, random_state=seed,
        )
        space = {
            "n_estimators": randint(200, 1501),
            "max_depth": randint(2, 11),
            "learning_rate": loguniform(0.005, 0.3),
            "subsample": uniform(0.5, 0.5),
            "colsample_bytree": uniform(0.5, 0.5),
            "min_child_weight": loguniform(0.1, 20.0),
            "gamma": loguniform(1e-8, 10.0),
            "reg_alpha": loguniform(1e-8, 10.0),
            "reg_lambda": loguniform(0.1, 50.0),
            "max_delta_step": [0, 1, 3, 5],
        }
        return est, space

    if model_key == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        est = lgb.LGBMClassifier(
            n_jobs=1, verbose=-1, class_weight="balanced", random_state=seed,
        )
        space = {
            "n_estimators": randint(200, 1501),
            "num_leaves": randint(15, 256),
            "learning_rate": loguniform(0.005, 0.3),
            "max_depth": [-1, 3, 4, 5, 6, 8, 10, 12, 16],
            "min_child_samples": randint(5, 101),
            "subsample": uniform(0.5, 0.5),
            "subsample_freq": [0, 1, 5],
            "colsample_bytree": uniform(0.5, 0.5),
            "reg_alpha": loguniform(1e-8, 10.0),
            "reg_lambda": loguniform(1e-8, 50.0),
            "min_split_gain": loguniform(1e-8, 1.0),
        }
        return est, space

    if model_key == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            return None
        est = CatBoostClassifier(
            verbose=False, allow_writing_files=False,
            auto_class_weights="Balanced", random_seed=seed,
        )
        space = {
            "iterations": randint(200, 1501),
            "depth": randint(3, 11),
            "learning_rate": loguniform(0.005, 0.3),
            "l2_leaf_reg": loguniform(1e-2, 30.0),
            "random_strength": loguniform(1e-3, 10.0),
            "bagging_temperature": uniform(0.0, 1.0),
            "border_count": [32, 64, 128, 254],
            "grow_policy": ["SymmetricTree", "Depthwise"],
            "leaf_estimation_iterations": randint(1, 11),
        }
        return est, space

    raise ValueError(f"Unknown model key: {model_key!r}")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Make column names safe for LightGBM (no JSON-special chars)."""
    safe = [str(c).replace(",", "_").replace("[", "_").replace("]", "_")
            .replace("<", "_").replace(">", "_").replace(":", "_").replace('"', "_")
            for c in X.columns]
    if list(X.columns) == safe:
        return X
    return X.rename(columns=dict(zip(X.columns, safe)))


def _load_train(dataset_name: str, logger: logging.Logger) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Prefer precomputed splits, fall back to the full processed CSV.

    Search is fit on train only; the held-out test set is left alone
    so it stays available for ``run_experiments.py`` to evaluate final
    models with the chosen parameters.
    """
    split = load_precomputed_split(dataset_name, logger=logger)
    if split is not None:
        X_train, _X_test, y_train, _y_test = split
        return _sanitize_columns(X_train), y_train

    path = GRID_DATASETS.get(dataset_name)
    if path is None or not path.exists():
        logger.error(f"Dataset {dataset_name!r}: no split and no processed CSV at {path}")
        return None
    X, y = load_dataset_from_csv(path)
    X = sanitize_feature_columns(coerce_features(X))
    return _sanitize_columns(X), y


def _params_to_key(params: Dict[str, Any]) -> str:
    """Deterministic JSON key for a parameter dict (used to align across seeds)."""
    return json.dumps(params, sort_keys=True, default=str)


def _sample_param_grid(
    model_key: str,
    param_space: Dict[str, Any],
    n_iter: int,
) -> List[Dict[str, List[Any]]]:
    """Sample ``n_iter`` random candidates and convert them to GridSearchCV form.

    We deliberately sample once per model with a deterministic seed and then
    reuse the same candidates for every dataset/seed. That makes the universal
    ranking statistically meaningful because every dataset evaluated the same
    candidate parameter sets.
    """
    model_offset = CLASSICAL_MODELS.index(model_key) + 1
    sample_seed = RANDOM_STATE + 1009 * model_offset
    sampled = list(ParameterSampler(
        param_space,
        n_iter=n_iter,
        random_state=sample_seed,
    ))
    return [{k: [v] for k, v in params.items()} for params in sampled]


def _cv_results_to_rows(
    cv_results: Dict[str, Any],
    *, dataset: str, model: str, seed: int,
) -> List[Dict[str, Any]]:
    """Flatten search ``cv_results_`` to one row per sampled parameter combo."""
    rows: List[Dict[str, Any]] = []
    n = len(cv_results["params"])
    for i in range(n):
        params = cv_results["params"][i]
        row: Dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "params": _params_to_key(params),
            "mean_pr_auc": float(cv_results[f"mean_test_pr_auc"][i]),
            "std_pr_auc": float(cv_results[f"std_test_pr_auc"][i]),
            "mean_balanced_accuracy": float(cv_results[f"mean_test_balanced_accuracy"][i]),
            "std_balanced_accuracy": float(cv_results[f"std_test_balanced_accuracy"][i]),
            "fit_time_seconds": float(cv_results["mean_fit_time"][i]),
        }
        rows.append(row)
    return rows


def _load_existing_results(logger: logging.Logger) -> pd.DataFrame:
    """Load previous full results for ``--resume`` if they exist."""
    if not OUT_FULL.exists():
        return pd.DataFrame()
    try:
        existing = pd.read_csv(OUT_FULL)
    except Exception as exc:
        logger.warning(f"Could not read existing results for resume: {exc}")
        return pd.DataFrame()

    required = {"dataset", "model", "seed", "params"}
    if not required.issubset(existing.columns):
        logger.warning(
            f"Existing {OUT_FULL.name} is missing resume columns; starting fresh"
        )
        return pd.DataFrame()
    logger.info(f"Resume loaded {len(existing)} existing rows from {OUT_FULL.name}")
    return existing


def _completed_pairs(
    existing: pd.DataFrame,
    seeds: Iterable[int],
    n_iter: int,
) -> set[Tuple[str, str]]:
    """Return (dataset, model) pairs that already have all expected rows."""
    if existing.empty:
        return set()

    expected_rows = len(list(seeds)) * int(n_iter)
    counts = (
        existing.drop_duplicates(["dataset", "model", "seed", "params"])
        .groupby(["dataset", "model"])
        .size()
        .reset_index(name="n_rows")
    )
    done = counts[counts["n_rows"] >= expected_rows]
    return set(zip(done["dataset"], done["model"]))


# ── Core randomized search ──────────────────────────────────────────────────

def _run_random_search(
    model_key: str,
    dataset_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seeds: Iterable[int],
    n_iter: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Run one model on one dataset across all seeds.

    Internally this is randomized candidate sampling + GridSearchCV over those
    fixed candidates, which gives RandomizedSearchCV-like coverage while
    preserving identical candidate sets across datasets.
    """
    seeds_list = list(seeds)
    # Build once only to get the param space. The estimator is rebuilt per seed
    # below so estimator-level randomness follows the requested seed.
    initial = _build_estimator_and_space(model_key, y_train, seed=seeds_list[0])
    if initial is None:
        logger.warning(f"[{dataset_name}] {model_key}: dependency missing, skipping")
        return []
    _initial_estimator, param_space = initial
    param_grid = _sample_param_grid(model_key, param_space, n_iter)
    logger.info(
        f"[{dataset_name}] {model_key}: {len(param_grid)} random candidates "
        f"x {CV_FOLDS} folds x {len(seeds_list)} seeds"
    )

    rows: List[Dict[str, Any]] = []
    for seed in seeds_list:
        built = _build_estimator_and_space(model_key, y_train, seed=seed)
        if built is None:
            logger.warning(f"[{dataset_name}] {model_key}: dependency missing, skipping")
            return rows
        estimator, _param_space = built
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=SCORING,
            refit=PRIMARY_METRIC,
            cv=cv,
            n_jobs=-1,
            error_score=np.nan,
            return_train_score=False,
        )
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info(
            f"  seed={seed:>4d}  best PR-AUC={gs.best_score_:.4f}  "
            f"elapsed={elapsed:.1f}s  best={gs.best_params_}"
        )
        rows.extend(_cv_results_to_rows(
            gs.cv_results_, dataset=dataset_name, model=model_key, seed=seed,
        ))
    return rows


# ── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate_per_dataset(full: pd.DataFrame) -> pd.DataFrame:
    """For every (dataset, model, params) compute mean across seeds and pick top.

    Returns the *best* parameter combo per (dataset, model) ranked by
    mean PR-AUC across seeds. Ties broken by lower std (more stable).
    """
    agg = (
        full.groupby(["dataset", "model", "params"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_pr_auc=("mean_pr_auc", "mean"),
            std_pr_auc=("mean_pr_auc", "std"),
            cv_std_pr_auc=("std_pr_auc", "mean"),
            mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
            mean_fit_time=("fit_time_seconds", "mean"),
        )
    )
    agg["std_pr_auc"] = agg["std_pr_auc"].fillna(0.0)
    agg = agg.sort_values(
        ["dataset", "model", "mean_pr_auc", "std_pr_auc"],
        ascending=[True, True, False, True],
    )
    top = agg.groupby(["dataset", "model"], as_index=False).head(1)
    return top.reset_index(drop=True)


def _aggregate_universal(full: pd.DataFrame) -> pd.DataFrame:
    """For every model, pick the parameter combo that ranks best across datasets.

    Strategy
    --------
    1. Mean PR-AUC across seeds within each (dataset, model, params).
    2. Within each (dataset, model), rank parameter combos by descending
       PR-AUC (best = rank 1).
    3. Average ranks across datasets — this normalises away score-scale
       differences between datasets (Taiwan ≈ 0.4-0.6, Polish ≈ 0.9+).
    4. Combo with the smallest mean rank is the universal pick; ties broken
       by higher mean PR-AUC and lower worst-case PR-AUC across datasets.
    """
    per_dataset_mean = (
        full.groupby(["dataset", "model", "params"], as_index=False)
        .agg(
            mean_pr_auc=("mean_pr_auc", "mean"),
            mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
        )
    )

    per_dataset_mean["rank_in_dataset"] = (
        per_dataset_mean.groupby(["dataset", "model"])["mean_pr_auc"]
        .rank(ascending=False, method="min")
    )

    # Keep only parameter combos that appear in *all* datasets for the model
    # so we are comparing like with like.
    coverage = (
        per_dataset_mean.groupby(["model", "params"])["dataset"]
        .nunique()
        .reset_index(name="n_datasets")
    )
    n_total_datasets = per_dataset_mean["dataset"].nunique()
    coverage_full = coverage[coverage["n_datasets"] == n_total_datasets]
    aligned = per_dataset_mean.merge(coverage_full[["model", "params"]], on=["model", "params"])

    universal = (
        aligned.groupby(["model", "params"], as_index=False)
        .agg(
            mean_rank=("rank_in_dataset", "mean"),
            std_rank=("rank_in_dataset", "std"),
            mean_pr_auc=("mean_pr_auc", "mean"),
            min_pr_auc=("mean_pr_auc", "min"),
            max_pr_auc=("mean_pr_auc", "max"),
            mean_balanced_accuracy=("mean_balanced_accuracy", "mean"),
        )
    )
    universal["std_rank"] = universal["std_rank"].fillna(0.0)
    universal = universal.sort_values(
        ["model", "mean_rank", "mean_pr_auc", "min_pr_auc"],
        ascending=[True, True, False, False],
    )

    top = universal.groupby("model", as_index=False).head(1)
    return top.reset_index(drop=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Random-search the 8 classical models across datasets.")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Subset of datasets (default: all three).")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Subset of models. Choices: {CLASSICAL_MODELS}")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help=f"Random seeds (default: {SEEDS}).")
    p.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER,
                   help=f"Random parameter candidates per model (default: {DEFAULT_N_ITER}).")
    p.add_argument("--resume", action="store_true",
                   help="Reuse random_search_full_results.csv and skip completed (dataset, model) pairs.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    ensure_output_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("random_search")

    datasets = list(args.datasets) if args.datasets else list(GRID_DATASETS.keys())
    models = list(args.models) if args.models else list(CLASSICAL_MODELS)
    seeds = list(args.seeds) if args.seeds else list(SEEDS)
    n_iter = int(args.n_iter)
    existing = _load_existing_results(logger) if args.resume else pd.DataFrame()
    completed = _completed_pairs(existing, seeds=seeds, n_iter=n_iter)
    if args.resume and not existing.empty:
        completed_index = pd.MultiIndex.from_tuples(
            completed, names=["dataset", "model"]
        )
        row_index = pd.MultiIndex.from_frame(existing[["dataset", "model"]])
        keep_mask = row_index.isin(completed_index)
        dropped = int((~keep_mask).sum())
        existing = existing.loc[keep_mask].copy()
        if dropped:
            logger.info(
                f"Resume ignored {dropped} rows from incomplete pairs; "
                "those pairs will be recomputed"
            )

    logger.info("=" * 80)
    logger.info("RANDOM SEARCH PIPELINE")
    logger.info(f"  datasets : {datasets}")
    logger.info(f"  models   : {models}")
    logger.info(f"  seeds    : {seeds}")
    logger.info(f"  n_iter   : {n_iter} random candidates per model")
    logger.info(f"  resume   : {'yes' if args.resume else 'no'}")
    logger.info(f"  scoring  : {SCORING} (refit on {PRIMARY_METRIC!r})")
    logger.info(f"  cv folds : {CV_FOLDS}  (StratifiedKFold, shuffle=True)")
    logger.info("=" * 80)

    all_rows: List[Dict[str, Any]] = existing.to_dict("records") if args.resume else []
    started = time.time()

    for dataset_name in datasets:
        if dataset_name not in DATASETS and dataset_name not in GRID_DATASETS:
            logger.warning(f"Dataset {dataset_name!r} not registered; skipping")
            continue
        loaded = _load_train(dataset_name, logger=logger)
        if loaded is None:
            continue
        X_train, y_train = loaded
        y_arr = np.asarray(y_train)
        pos = int((y_arr == 1).sum())
        neg = int((y_arr != 1).sum())
        logger.info(
            f"\n>>> dataset={dataset_name}  shape={X_train.shape}  "
            f"positives={pos}  negatives={neg}  rate={pos / max(pos + neg, 1):.3%}"
        )

        for model_key in models:
            if model_key not in CLASSICAL_MODELS:
                logger.warning(f"Skipping non-classical model: {model_key!r}")
                continue
            if (dataset_name, model_key) in completed:
                logger.info(f"[{dataset_name}] {model_key}: already complete, skipping")
                continue
            try:
                rows = _run_random_search(
                    model_key=model_key,
                    dataset_name=dataset_name,
                    X_train=X_train,
                    y_train=y_train,
                    seeds=seeds,
                    n_iter=n_iter,
                    logger=logger,
                )
                all_rows.extend(rows)
                # Persist after each (dataset, model) so a crash leaves usable data.
                pd.DataFrame(all_rows).to_csv(OUT_FULL, index=False)
            except Exception as exc:
                logger.error(
                    f"[{dataset_name}] {model_key}: random search failed: {exc}",
                    exc_info=True,
                )

    if not all_rows:
        logger.error("No random-search rows produced. Nothing to aggregate.")
        return 1

    full = pd.DataFrame(all_rows)
    full.to_csv(OUT_FULL, index=False)
    logger.info(f"\nSaved {OUT_FULL.name}: {len(full)} rows")

    top_per_dataset = _aggregate_per_dataset(full)
    top_per_dataset.to_csv(OUT_TOP_PER_DATASET, index=False)
    logger.info(f"Saved {OUT_TOP_PER_DATASET.name}: {len(top_per_dataset)} rows")

    universal = _aggregate_universal(full)
    universal.to_csv(OUT_UNIVERSAL, index=False)
    logger.info(f"Saved {OUT_UNIVERSAL.name}: {len(universal)} rows")

    logger.info("\nTop-1 params per (dataset, model):")
    for _, r in top_per_dataset.iterrows():
        logger.info(
            f"  {r['dataset']:24s} {r['model']:20s} "
            f"pr_auc={r['mean_pr_auc']:.4f}  bacc={r['mean_balanced_accuracy']:.4f}  "
            f"params={r['params']}"
        )

    logger.info("\nUniversal best params per model (lowest mean rank across datasets):")
    for _, r in universal.iterrows():
        logger.info(
            f"  {r['model']:20s} "
            f"mean_rank={r['mean_rank']:.2f}  pr_auc(mean/min)="
            f"{r['mean_pr_auc']:.4f}/{r['min_pr_auc']:.4f}  "
            f"params={r['params']}"
        )

    logger.info(f"\nDone in {time.time() - started:.1f}s. Output dir: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
