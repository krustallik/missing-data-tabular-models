"""Generate the three markdown reports from consolidated results.

Outputs:
- ``data_methods_report.md`` — datasets, split, metrics, missingness logic,
  imputation methods, model roster.
- ``results_discussion_report.md`` — tables + discussion of what works under
  which (mechanism, rate, imputation) combinations.
- ``practical_usability_report.md`` + ``deployment_guide.md`` +
  ``deployment_complexity.csv`` + ``suitability_matrix.csv`` —
  practical recommendations for real-world usage.
- ``interpretation_guide.md`` / ``presentation_points.txt`` — summaries
  suitable for presentations / talking points.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from config import (
    DATASETS,
    IMPUTATION_METHODS,
    MISSING_MECHANISMS,
    MISSING_RATES,
    OUTPUT_FILES,
    PERFORMANCE_ACCEPTABLE,
    PERFORMANCE_EXCELLENT,
    PERFORMANCE_GOOD,
    PRIMARY_METRICS,
    REPORT_AUTHOR,
    REPORT_INSTITUTION,
    REPORT_TITLE,
    ensure_output_dirs,
)
from data_utils import setup_logging
from models import model_type as _resolve_model_type


# ── Markdown helpers ─────────────────────────────────────────────────────────

def _df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """Minimal Markdown table renderer without requiring ``tabulate``."""
    if df is None or df.empty:
        return "_(no data)_"
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, r in df.iterrows():
        parts = []
        for v in r:
            if isinstance(v, float):
                parts.append("" if pd.isna(v) else format(v, float_fmt))
            elif v is None or (isinstance(v, float) and pd.isna(v)):
                parts.append("")
            else:
                parts.append(str(v))
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines)


def _load(logger: logging.Logger) -> Optional[pd.DataFrame]:
    path = OUTPUT_FILES["consolidated_results"]
    if not path.exists():
        path = OUTPUT_FILES["experiment_results"]
    if not path.exists():
        logger.error("Neither consolidated_results.csv nor experiment_results.csv present.")
        return None
    df = pd.read_csv(path)
    if "model" in df.columns:
        df["model_type"] = df["model"].map(_resolve_model_type)
    logger.info(f"Loaded {path.name}: {len(df)} rows")
    return df


# ── Data / Methods report ────────────────────────────────────────────────────

def _data_methods_report(df: Optional[pd.DataFrame]) -> str:
    lines: List[str] = [
        f"# {REPORT_TITLE}",
        "",
        f"**Author**: {REPORT_AUTHOR}  ",
        f"**Institution**: {REPORT_INSTITUTION}  ",
        f"**Date**: {datetime.now().strftime('%B %d, %Y')}",
        "",
        "This document covers the Data, Experimental setup and Methods "
        "chapters. Results and Discussion live in `results_discussion_report.md` "
        "and practical recommendations in `practical_usability_report.md`.",
        "",
        "---",
        "",
        "## 1. Data",
        "",
        "Three real-world tabular classification datasets from the course "
        "reference article are used:",
        "",
    ]
    lines.append(
        "- **Taiwan Bankruptcy** - financial ratios for Taiwanese listed firms.\n"
        "- **Polish Companies (1-year horizon)** - Polish bankruptcy forecasting.\n"
        "- **Slovak Manufacture 13** - Slovak SME financial indicators.\n"
    )
    lines.append(
        "All three are stored pre-cleaned in `data/processed/` with a single "
        "`target` column. Feature matrices are originally complete (no native "
        "NaN) which is the precondition for the controlled missingness protocol."
    )

    if df is not None and not df.empty:
        lines += [
            "",
            "### 1.1 Coverage in the consolidated table",
            "",
        ]
        meta = df.groupby("dataset").agg(
            n_rows=("model", "size"),
            models=("model", lambda s: ", ".join(sorted(set(s.dropna())))),
        ).reset_index()
        lines.append(_df_to_markdown(meta))
    lines += [
        "",
        "---",
        "",
        "## 2. Experimental setup",
        "",
        "- **Split**: stratified 80/20 train/test split, `random_state = 42`. "
        "Splits are materialised once in `data/splits/*.csv` and re-used by "
        "every step of the benchmark.",
        f"- **Primary metrics** (ranking): {', '.join(PRIMARY_METRICS)}. "
        "On highly imbalanced binary targets `accuracy` is misleading — "
        "`balanced_accuracy`, `f1_macro` and `pr_auc` are the ones to trust. "
        "Additional metrics stored per run: `f1` (weighted), `precision`, "
        "`recall`, `recall_class1`, `roc_auc`, plus the decision `threshold` "
        "actually used after tuning.",
        f"- **Missingness grid**: mechanisms {MISSING_MECHANISMS}, rates "
        f"{[f'{int(r*100)}%' for r in MISSING_RATES]}. Injection is applied "
        "to the **training split only**; the test split is kept complete so "
        "reported metrics measure the effect of training under missingness.",
        "- **Verification**: realised missing fractions are compared against "
        "requested rates in `results/tables/missingness_verification.csv`. "
        "The injector stays within ±1 percentage point of target.",
        "",
        "---",
        "",
        "## 3. Methods",
        "",
        "### 3.1 Missingness generation",
        "",
        "- **MCAR**: uniform random sampling of observed cells.\n"
        "- **MAR**: probability of masking a cell in feature *A* depends on a "
        "control feature *B* (rows where *B* exceeds its median receive weight "
        "0.8, others 0.2).\n"
        "- **MNAR**: probability of masking a cell depends on the feature's own "
        "value.\n",
        "Implementation: `src/missingness.py`.",
        "",
        "### 3.2 Imputation methods",
        "",
        "All methods fit on train only to avoid test leakage:",
        "",
        "- **mean / median** - per-column train statistic.\n"
        "- **knn** - `KNNImputer(n_neighbors=5)`.\n"
        "- **mice** - `IterativeImputer` (MICE).\n"
        "- **mice_indicator** - MICE + binary missing-indicator features.\n"
        "- **none** - raw NaN passed to models that handle NaN natively "
        "(TabPFN, TabICL, CatBoost, XGBoost, LightGBM).\n",
        "Implementation: `src/imputation.py`.",
        "",
        "### 3.3 Models",
        "",
        "Classical: Logistic Regression, Random Forest, Gradient Boosting, "
        "XGBoost, LightGBM, SVM, MLP, CatBoost (default hyperparameters; "
        "scaling applied to LR / SVM / MLP).",
        "",
        "Foundation: TabPFN, TabICL. TabPFN requires `TABPFN_TOKEN` "
        "and skips gracefully when it is absent. GPU is used when available.",
        "",
        "Implementation: `src/models.py`.",
        "",
        "### 3.4 Evaluation",
        "",
        f"Metrics are computed in `src/evaluation.py` using scikit-learn's "
        f"weighted averages for multi-class-safe F1/Precision/Recall and "
        f"binary ROC-AUC (NaN for multiclass targets).",
        "",
    ]
    return "\n".join(lines) + "\n"


# ── Results / Discussion report ──────────────────────────────────────────────

def _classify_performance(acc: float) -> str:
    if pd.isna(acc):
        return "n/a"
    if acc >= PERFORMANCE_EXCELLENT:
        return "Excellent"
    if acc >= PERFORMANCE_GOOD:
        return "Good"
    if acc >= PERFORMANCE_ACCEPTABLE:
        return "Acceptable"
    return "Below target"


def _results_discussion_report(df: pd.DataFrame) -> str:
    lines = [
        f"# {REPORT_TITLE} - Results and Discussion",
        "",
        f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "This document reports the numerical outcomes of the experiment "
        "matrix (datasets x mechanisms x rates x imputations x models) "
        "consolidated in `consolidated_results.csv`.",
        "",
        "---",
        "",
        "## 1. Summary",
        "",
    ]
    succeeded = df.dropna(subset=["accuracy"])
    failed = df[df["accuracy"].isna()]

    has_bacc = "balanced_accuracy" in succeeded.columns
    has_f1m = "f1_macro" in succeeded.columns
    has_prauc = "pr_auc" in succeeded.columns
    has_rc1 = "recall_class1" in succeeded.columns
    ranking_metric = "balanced_accuracy" if has_bacc else "accuracy"

    lines.append(
        f"- Total runs recorded: **{len(df)}**\n"
        f"- Successful runs:      **{len(succeeded)}**\n"
        f"- Failed / skipped:     **{len(failed)}**\n"
    )
    if not succeeded.empty:
        lines.append(f"- Unique models: **{succeeded['model'].nunique()}**\n")
        lines.append(
            f"- Mean **accuracy** across successful runs: "
            f"**{succeeded['accuracy'].mean():.4f}**  "
            f"_(misleading on imbalanced data — see balanced_accuracy below)_\n"
        )
        if has_bacc:
            lines.append(
                f"- Mean **balanced_accuracy**: "
                f"**{succeeded['balanced_accuracy'].mean():.4f}**  "
                f"_(honest summary — 0.5 = random on binary tasks)_\n"
            )
        if has_prauc:
            lines.append(
                f"- Mean **pr_auc** (binary tasks): "
                f"**{succeeded['pr_auc'].mean(skipna=True):.4f}**  "
                f"_(how well the positive class is ranked above negatives)_\n"
            )
        if has_rc1:
            lines.append(
                f"- Mean **recall_class1**: "
                f"**{succeeded['recall_class1'].mean(skipna=True):.4f}**  "
                f"_(fraction of true positives actually caught)_\n"
            )

    lines += [
        "",
        "## 2. Best configuration per model",
        "",
        f"Mean metrics per model (averaged over datasets and scenarios), "
        f"ranked by **{ranking_metric}** — the honest summary on imbalanced data. "
        f"`accuracy` is kept for comparison but is not the primary ranking.",
        "",
    ]
    if not succeeded.empty:
        agg_spec = {
            "accuracy": ("accuracy", "mean"),
            "f1_weighted": ("f1", "mean"),
            "runs": ("accuracy", "size"),
        }
        if has_bacc:
            agg_spec["balanced_accuracy"] = ("balanced_accuracy", "mean")
        if has_f1m:
            agg_spec["f1_macro"] = ("f1_macro", "mean")
        if has_prauc:
            agg_spec["pr_auc"] = ("pr_auc", "mean")
        if has_rc1:
            agg_spec["recall_class1"] = ("recall_class1", "mean")

        best = (
            succeeded.groupby("model")
            .agg(**agg_spec)
            .round(4)
            .sort_values(ranking_metric, ascending=False)
            .reset_index()
        )
        best["performance"] = best["accuracy"].map(_classify_performance)
        lines.append(_df_to_markdown(best))
    lines += [
        "",
        "## 3. Classical vs Foundation",
        "",
    ]
    if not succeeded.empty and "model_type" in succeeded.columns:
        agg_cmp = {
            "accuracy_mean": ("accuracy", "mean"),
            "accuracy_std": ("accuracy", "std"),
            "f1_weighted_mean": ("f1", "mean"),
            "n": ("accuracy", "size"),
        }
        if has_bacc:
            agg_cmp["balanced_accuracy_mean"] = ("balanced_accuracy", "mean")
            agg_cmp["balanced_accuracy_std"] = ("balanced_accuracy", "std")
        if has_f1m:
            agg_cmp["f1_macro_mean"] = ("f1_macro", "mean")
        if has_prauc:
            agg_cmp["pr_auc_mean"] = ("pr_auc", "mean")

        cmp = (
            succeeded.groupby("model_type")
            .agg(**agg_cmp)
            .round(4)
            .reset_index()
        )
        lines.append(_df_to_markdown(cmp))

    lines += [
        "",
        "## 4. Robustness across missingness mechanisms",
        "",
        f"Mean **{ranking_metric}** per (model, mechanism) across all rates "
        f"and imputations:",
        "",
    ]
    rob = succeeded.dropna(subset=["missing_mechanism"])
    if not rob.empty:
        grp = (
            rob.groupby(["model", "missing_mechanism"])[ranking_metric]
            .mean()
            .unstack(fill_value=float("nan"))
            .round(4)
            .reset_index()
        )
        lines.append(_df_to_markdown(grp))

    lines += [
        "",
        f"## 5. {ranking_metric.replace('_', ' ').title()} vs missing rate",
        "",
        f"Mean **{ranking_metric}** per (model, missing_rate) averaged over "
        f"mechanisms, imputations, and datasets:",
        "",
    ]
    if not succeeded.empty and "missing_rate" in succeeded.columns:
        rate_pivot = (
            succeeded.dropna(subset=["missing_rate"])
            .groupby(["model", "missing_rate"])[ranking_metric]
            .mean()
            .unstack(fill_value=float("nan"))
            .round(4)
            .reset_index()
        )
        lines.append(_df_to_markdown(rate_pivot))

    lines += [
        "",
        "## 6. Imputation method ranking",
        "",
        f"Mean **{ranking_metric}** averaged across models / datasets / "
        f"mechanisms / rates:",
        "",
    ]
    imp = (
        succeeded.groupby("imputation")[ranking_metric]
        .agg(["mean", "std", "count"])
        .round(4)
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    if not imp.empty:
        lines.append(_df_to_markdown(imp))

    lines += [
        "",
        "## 7. Discussion",
        "",
        "- **Foundation models vs classical baselines**: see table in §3 and "
        "the boxplot in `visualizations/classical_vs_foundation.png`.\n"
        "- **Imputation sensitivity**: §6 ranks imputation methods by mean "
        "accuracy; `mice_indicator` is expected to shine with higher rates "
        "while simple mean/median are competitive at low rates.\n"
        "- **Missingness mechanism**: MNAR is the hardest scenario (value "
        "itself drives missingness), so gaps between mechanisms in §4 are "
        "informative for real-world applicability.\n",
        "",
        "## 8. Conclusion",
        "",
        "The consolidated table produced by `experiment_runner.py` gives a "
        "single source of truth for answering the project's question - which "
        "combination of (imputation, model) is most robust across the "
        "MCAR/MAR/MNAR grid. Practical recommendations follow in "
        "`practical_usability_report.md`.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ── Practical usability ──────────────────────────────────────────────────────

_DEPLOYMENT_ROWS = [
    {
        "model": "Logistic-Regression",
        "install": "sklearn only",
        "gpu_required": "no",
        "typical_training_seconds": "<1",
        "size_limits": "scales linearly; no practical cap",
        "notes": "Needs imputation and scaling; weakest classical baseline on nonlinear data.",
    },
    {
        "model": "Random-Forest",
        "install": "sklearn only",
        "gpu_required": "no",
        "typical_training_seconds": "1-30",
        "size_limits": "millions of rows ok with n_jobs=-1",
        "notes": "Robust default; does not need scaling; needs imputation.",
    },
    {
        "model": "Gradient-Boosting",
        "install": "sklearn only",
        "gpu_required": "no",
        "typical_training_seconds": "5-60",
        "size_limits": "up to ~100k rows comfortably",
        "notes": "Sensitive to hyperparameters; needs imputation.",
    },
    {
        "model": "XGBoost",
        "install": "pip install xgboost",
        "gpu_required": "optional",
        "typical_training_seconds": "1-30",
        "size_limits": "very large (millions of rows)",
        "notes": "Handles NaN natively; strong default for tabular tasks.",
    },
    {
        "model": "LightGBM",
        "install": "pip install lightgbm",
        "gpu_required": "optional",
        "typical_training_seconds": "1-20",
        "size_limits": "very large (millions of rows)",
        "notes": "Handles NaN natively; fastest among boosted trees here.",
    },
    {
        "model": "SVM",
        "install": "sklearn only",
        "gpu_required": "no",
        "typical_training_seconds": "10-300",
        "size_limits": "tens of thousands (O(n^2) kernel)",
        "notes": "Requires scaling and imputation; slow on large datasets.",
    },
    {
        "model": "MLP",
        "install": "sklearn only",
        "gpu_required": "no (torch-backed MLPs do)",
        "typical_training_seconds": "10-180",
        "size_limits": "hundreds of thousands",
        "notes": "Requires scaling and imputation; sensitive to architecture.",
    },
    {
        "model": "TabPFN",
        "install": "pip install tabpfn + TABPFN_TOKEN",
        "gpu_required": "recommended (CUDA)",
        "typical_training_seconds": "1-60 per call (pretrained, no training)",
        "size_limits": "up to ~10k rows, ~500 features per call",
        "notes": "Consumes raw NaN; requires license token; best for small-to-medium tabular.",
    },
    {
        "model": "TabICL",
        "install": "pip install tabicl",
        "gpu_required": "recommended (CUDA)",
        "typical_training_seconds": "5-120 per call",
        "size_limits": "similar to TabPFN; in-context learning",
        "notes": "Pretrained, handles NaN; verify availability in the target environment.",
    },
    {
        "model": "CatBoost",
        "install": "pip install catboost",
        "gpu_required": "optional",
        "typical_training_seconds": "10-120",
        "size_limits": "very large",
        "notes": "Handles NaN and categorical features natively; robust default.",
    },
]


def _deployment_complexity() -> pd.DataFrame:
    return pd.DataFrame(_DEPLOYMENT_ROWS)


def _suitability_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Score each model on (low/med/high) missingness scenarios.

    Uses ``balanced_accuracy`` when available (honest on imbalanced datasets);
    falls back to ``accuracy`` for old CSVs without that column.
    """
    metric = "balanced_accuracy" if "balanced_accuracy" in df.columns else "accuracy"
    sub = df.dropna(subset=[metric, "missing_rate"])
    if sub.empty:
        return pd.DataFrame()
    buckets = pd.cut(
        sub["missing_rate"],
        bins=[-1, 10, 20, 100],
        labels=["low (<=10%)", "medium (<=20%)", "high (>20%)"],
    )
    sub = sub.assign(bucket=buckets)
    out = (
        sub.groupby(["model", "bucket"], observed=False)[metric]
        .mean()
        .unstack()
        .round(4)
        .reset_index()
    )
    out.attrs["metric"] = metric
    return out


def _practical_usability_report(df: pd.DataFrame) -> str:
    lines = [
        f"# {REPORT_TITLE} - Practical Usability",
        "",
        f"_Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "This document translates the benchmark into practical guidance for "
        "real-world use: at what missingness level is each model usable, and "
        "what does it take to deploy it.",
        "",
        "---",
        "",
        "## 1. Suitability matrix",
        "",
    ]
    matrix = _suitability_matrix(df)
    metric_used = matrix.attrs.get("metric", "accuracy") if not matrix.empty else "accuracy"
    lines += [
        f"Mean **{metric_used}** by missingness bucket (all mechanisms, "
        f"all imputations). `balanced_accuracy` is used by default as it is "
        f"honest on imbalanced datasets (0.5 ≈ random guessing).",
        "",
    ]
    lines.append(_df_to_markdown(matrix))
    lines += [
        "",
        "## 2. Performance bands",
        "",
        f"- Excellent: accuracy >= {PERFORMANCE_EXCELLENT:.2f}\n"
        f"- Good:      accuracy >= {PERFORMANCE_GOOD:.2f}\n"
        f"- Acceptable: accuracy >= {PERFORMANCE_ACCEPTABLE:.2f}\n",
        "",
        "## 3. Foundation models - when to use",
        "",
        "- **TabPFN** is strongest for small-to-medium datasets where labeled "
        "data is scarce. It does not need imputation and gives competitive "
        "accuracy at low-to-medium missingness rates. License token required.\n"
        "- **TabICL** competes with TabPFN on small/medium tabular tasks with "
        "the advantage of NaN handling; runtime and size limits depend on the "
        "particular implementation version.\n",
        "",
        "## 4. Foundation models - when NOT to use",
        "",
        "- Very large datasets (millions of rows) -> use XGBoost/LightGBM.\n"
        "- High feature counts above TabPFN's pretraining limit -> classical "
        "boosting is safer.\n"
        "- Environments without GPU / without license access -> CatBoost or "
        "XGBoost with MICE + indicator (CatBoost is the most robust NaN-aware "
        "classical tree ensemble and a good baseline when TabPFN / TabICL are "
        "not an option).\n",
        "",
        "## 5. Deployment complexity",
        "",
        "See also `deployment_complexity.csv` and `deployment_guide.md`.",
        "",
    ]
    lines.append(_df_to_markdown(_deployment_complexity()))
    return "\n".join(lines) + "\n"


def _deployment_guide() -> str:
    rows = _deployment_complexity()
    lines = [
        f"# Deployment Guide",
        "",
        "Quick-reference installation and operational notes for every model "
        "in the benchmark.",
        "",
    ]
    for _, r in rows.iterrows():
        lines += [
            f"## {r['model']}",
            "",
            f"- **Install**: `{r['install']}`",
            f"- **GPU required**: {r['gpu_required']}",
            f"- **Typical training time**: {r['typical_training_seconds']}",
            f"- **Size limits**: {r['size_limits']}",
            f"- **Notes**: {r['notes']}",
            "",
        ]
    return "\n".join(lines)


# ── Interpretation guide + presentation points ──────────────────────────────

def _interpretation_guide(df: Optional[pd.DataFrame]) -> str:
    lines = [
        "# Interpretation guide",
        "",
        "How to read the tables and plots produced by this pipeline.",
        "",
        "## Files in `results/tables/`",
        "",
        "- `dataset_overview.csv` - per-dataset size and class balance.",
        "- `experiment_setup.csv` - random seed, test fraction, metrics used.",
        "- `missingness_verification.csv` - realised vs target missing rates.",
        "- `imputation_benchmark.csv` - imputation time and errors per scenario.",
        "- `experiment_results.csv` - full benchmark table (one row per run).",
        "- `consolidated_results.csv` - same rows with the canonical column "
        "schema used by Results / Discussion.",
        "- `classical_models.csv` / `foundation_models.csv` - convenience splits.",
        "- `robustness_analysis.csv` - per-(model, mechanism) aggregates.",
        "",
        "## Files in `results/visualizations/`",
        "",
        "- `missing_rate_MCAR.png` / `missing_rate_MAR.png` / `missing_rate_MNAR.png` - "
        "primary metrics (accuracy, balanced_accuracy, f1_macro, pr_auc) as "
        "functions of missing rate for each mechanism.",
        "- `classical_vs_foundation.png` - distribution boxplot across the "
        "primary metrics.",
        "- `stability_heatmap.png` - balanced_accuracy heatmap across missing "
        "rates (the metric that actually reflects minority-class detection).",
        "- `per_dataset_ranking.png` - bar chart ranking on the native scenario "
        "by balanced_accuracy.",
        "",
        "## Files in `results/reports/`",
        "",
        "- `data_methods_report.md` - Data / Setup / Methods.",
        "- `results_discussion_report.md` - quantitative results + discussion.",
        "- `practical_usability_report.md` - practical recommendations.",
        "- `deployment_guide.md` - per-model install / GPU / size notes.",
        "- `deployment_complexity.csv` / `suitability_matrix.csv` - tables "
        "behind the practical report.",
        "",
    ]
    if df is not None and not df.empty:
        lines += [
            "## Quick numbers at hand",
            "",
            f"- Consolidated rows: {len(df)}",
            f"- Datasets: {', '.join(sorted(df['dataset'].dropna().unique()))}",
            f"- Models: {', '.join(sorted(df['model'].dropna().unique()))}",
        ]
    return "\n".join(lines) + "\n"


def _presentation_points(df: Optional[pd.DataFrame]) -> str:
    lines = [
        "Presentation points",
        "===================",
        "",
        "1. Shared experimental backbone: 3 datasets, stratified 80/20, seed 42.",
        "2. Controlled missingness: MCAR / MAR / MNAR at 5-40%.",
        "3. Imputations evaluated: mean, median, kNN, MICE, MICE+indicator, none.",
        "4. Classical models: Logistic Regression, Random Forest, Gradient Boosting, "
        "XGBoost, LightGBM, SVM, MLP, CatBoost.",
        "5. Foundation models: TabPFN, TabICL.",
        "6. Single consolidated CSV (`consolidated_results.csv`) with unified model names.",
        "",
        "Key questions answered by the tables:",
        "- Which models degrade fastest as missing rate grows?",
        "- Does MICE + indicator beat simple imputations at higher rates?",
        "- Do foundation models justify their deployment cost?",
        "",
    ]
    if df is not None and not df.empty:
        n = df["accuracy"].notna().sum()
        lines.append(
            f"Consolidated table: {len(df)} rows, {n} successful runs, "
            f"{df['model'].nunique()} unique models."
        )
    return "\n".join(lines) + "\n"


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_reports(logger: Optional[logging.Logger] = None) -> bool:
    ensure_output_dirs()
    if logger is None:
        logger = setup_logging("reporting")

    df = _load(logger)
    if df is None:
        return False

    OUTPUT_FILES["data_methods_report"].write_text(_data_methods_report(df), encoding="utf-8")
    logger.info(f"Saved {OUTPUT_FILES['data_methods_report'].name}")

    OUTPUT_FILES["results_discussion_report"].write_text(
        _results_discussion_report(df), encoding="utf-8",
    )
    logger.info(f"Saved {OUTPUT_FILES['results_discussion_report'].name}")

    OUTPUT_FILES["practical_usability_report"].write_text(
        _practical_usability_report(df), encoding="utf-8",
    )
    logger.info(f"Saved {OUTPUT_FILES['practical_usability_report'].name}")

    OUTPUT_FILES["deployment_guide"].write_text(_deployment_guide(), encoding="utf-8")
    logger.info(f"Saved {OUTPUT_FILES['deployment_guide'].name}")

    _deployment_complexity().to_csv(OUTPUT_FILES["deployment_complexity"], index=False)
    logger.info(f"Saved {OUTPUT_FILES['deployment_complexity'].name}")

    sm = _suitability_matrix(df)
    if not sm.empty:
        sm.to_csv(OUTPUT_FILES["suitability_matrix"], index=False)
        logger.info(f"Saved {OUTPUT_FILES['suitability_matrix'].name}")

    OUTPUT_FILES["interpretation_guide"].write_text(
        _interpretation_guide(df), encoding="utf-8",
    )
    logger.info(f"Saved {OUTPUT_FILES['interpretation_guide'].name}")

    OUTPUT_FILES["presentation_points"].write_text(
        _presentation_points(df), encoding="utf-8",
    )
    logger.info(f"Saved {OUTPUT_FILES['presentation_points'].name}")

    return True


if __name__ == "__main__":
    generate_reports()
