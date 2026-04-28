"""Generate the markdown reports from consolidated results.

Outputs:
- ``data_methods_report.md`` — datasets, split, metrics, missingness logic,
  imputation methods, model roster.
- ``results_discussion_report.md`` — tables + discussion of what works under
  which (mechanism, rate, imputation) combinations.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd

from config import (
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
        "chapters. Results and Discussion live in `results_discussion_report.md`.",
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
        "MCAR/MAR/MNAR grid.",
        "",
    ]
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

    return True


if __name__ == "__main__":
    generate_reports()
