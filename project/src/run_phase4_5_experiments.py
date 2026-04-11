"""Phase 4.5 - Final Analysis and Visualization.

Consolidates results from Phases 4.1-4.4 and generates comprehensive analysis:
- Unified comparison table of all models across datasets and preprocessing
- Visualizations: missing rate vs. model performance (MCAR/MAR/MNAR separate)
- Classical vs. Foundation model performance gaps
- Model stability and robustness analysis

Foundation models: TabICL, CatBoost (bonus)
Classical models:  Logistic Regression, Random Forest, SVM, MLP, XGBoost, LightGBM

Outputs:
- results/tables/phase4_5_consolidated_results.csv
- results/tables/phase4_5_classical_models.csv
- results/tables/phase4_5_foundation_models.csv
- results/tables/phase4_5_robustness_analysis.csv
- results/visualizations/phase4_5_missing_rate_MCAR.png
- results/visualizations/phase4_5_missing_rate_MAR.png
- results/visualizations/phase4_5_missing_rate_MNAR.png
- results/visualizations/phase4_5_model_comparison.png
- results/visualizations/phase4_5_stability_heatmap.png
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    plt = None
    sns = None

from config import RANDOM_STATE  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR    = PROJECT_ROOT / "results" / "logs"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
VIZ_DIR     = PROJECT_ROOT / "results" / "visualizations"

# ── Model classification ──────────────────────────────────────────────────────
FOUNDATION_MODEL_NAMES = {"TabICL", "CatBoost"}   # CatBoost = bonus foundation
CLASSICAL_MODEL_NAMES  = {
    "Logistic-Regression", "Random-Forest", "Svm", "Mlp", "Xgboost", "Lightgbm",
}

MISSINGNESS_PATTERNS = ["MCAR", "MAR", "MNAR"]
MISSING_RATES        = [5, 10, 15, 20, 30, 40]
PLOT_DPI             = 150
PLOT_STYLE           = "seaborn-v0_8-darkgrid"


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(name: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = logging.FileHandler(LOGS_DIR / f"experiment_{name}_{ts}.log")
    h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    return logger


def _load_json(path: Path, logger: logging.Logger) -> Optional[List[Dict]]:
    if not path.exists():
        logger.warning(f"File not found: {path.name}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info(f"Loaded {path.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading {path.name}: {e}")
        return None


def _model_type(model_name: str) -> str:
    return "Foundation" if model_name in FOUNDATION_MODEL_NAMES else "Classical"


# ── Phase 4.1 & 4.2 parser ───────────────────────────────────────────────────

def _parse_41_42(results: List[Dict], phase: float) -> List[Dict]:
    """Parse phase4_experiment_results.json and phase4_2_experiment_results.json."""
    rows = []
    for ds in results:
        dataset = ds.get("dataset")
        for prep, prep_data in ds.get("preprocessing", {}).items():
            for model_raw, model_data in prep_data.get("models", {}).items():
                if not isinstance(model_data, dict) or "test_metrics" not in model_data:
                    continue
                m = model_data["test_metrics"]
                model = model_raw.replace("_", "-").title()
                rows.append({
                    "phase": phase,
                    "dataset": dataset,
                    "model": model,
                    "model_type": _model_type(model),
                    "preprocessing": prep,
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "roc_auc": m.get("roc_auc"),
                    "training_time_seconds": model_data.get("training_time_seconds"),
                    "missing_mechanism": None,
                    "missing_rate": None,
                })
    return rows


# ── Phase 4.3 parser ─────────────────────────────────────────────────────────

def _parse_43(results: List[Dict]) -> List[Dict]:
    """Parse phase4_3_gradient_boosting_results.json."""
    rows = []
    for ds in results:
        dataset = ds.get("dataset")
        for scenario_key, scenario_data in ds.get("scenarios", {}).items():
            # scenario_key = "MCAR_5pct"
            try:
                parts = scenario_key.split("_")
                mechanism = parts[0]
                missing_rate = int(parts[1].replace("pct", ""))
            except Exception:
                mechanism, missing_rate = "Unknown", 0

            for prep, prep_data in scenario_data.items():
                for model_raw, model_metrics in prep_data.get("models", {}).items():
                    if not isinstance(model_metrics, dict):
                        continue
                    if "error" in model_metrics:
                        continue
                    model = model_raw.replace("_", "-").title()
                    rows.append({
                        "phase": 4.3,
                        "dataset": dataset,
                        "model": model,
                        "model_type": _model_type(model),
                        "preprocessing": prep,
                        "accuracy": model_metrics.get("accuracy"),
                        "f1": model_metrics.get("f1"),
                        "precision": model_metrics.get("precision"),
                        "recall": model_metrics.get("recall"),
                        "roc_auc": model_metrics.get("roc_auc"),
                        "training_time_seconds": model_metrics.get("training_time_seconds"),
                        "missing_mechanism": mechanism,
                        "missing_rate": missing_rate,
                    })
    return rows


# ── Phase 4.4 parser (TabICL + CatBoost) ─────────────────────────────────────

def _parse_44_tabicl(results: List[Dict]) -> List[Dict]:
    """Parse phase4_4_tabicl_results.json — contains TabICL and CatBoost."""
    rows = []
    for ds in results:
        dataset = ds.get("dataset")

        # TabICL with imputation
        for prep, r in ds.get("tabicl_with_imputation", {}).items():
            if r.get("available") and r.get("metrics"):
                m = r["metrics"]
                rows.append({
                    "phase": 4.4,
                    "dataset": dataset,
                    "model": "TabICL",
                    "model_type": "Foundation",
                    "preprocessing": prep,
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "roc_auc": m.get("roc_auc"),
                    "training_time_seconds": r.get("training_time_seconds"),
                    "missing_mechanism": None,
                    "missing_rate": None,
                })

        # TabICL without imputation (native NaN)
        for prep, r in ds.get("tabicl_without_imputation", {}).items():
            if r.get("available") and r.get("metrics"):
                m = r["metrics"]
                rows.append({
                    "phase": 4.4,
                    "dataset": dataset,
                    "model": "TabICL",
                    "model_type": "Foundation",
                    "preprocessing": "raw_nan",
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "roc_auc": m.get("roc_auc"),
                    "training_time_seconds": r.get("training_time_seconds"),
                    "missing_mechanism": None,
                    "missing_rate": None,
                })

        # CatBoost with imputation (bonus)
        for prep, r in ds.get("catboost_with_imputation", {}).items():
            if r.get("available") and r.get("metrics"):
                m = r["metrics"]
                rows.append({
                    "phase": 4.4,
                    "dataset": dataset,
                    "model": "CatBoost",
                    "model_type": "Foundation",
                    "preprocessing": prep,
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "roc_auc": m.get("roc_auc"),
                    "training_time_seconds": r.get("training_time_seconds"),
                    "missing_mechanism": None,
                    "missing_rate": None,
                })

        # CatBoost without imputation (bonus)
        for prep, r in ds.get("catboost_without_imputation", {}).items():
            if r.get("available") and r.get("metrics"):
                m = r["metrics"]
                rows.append({
                    "phase": 4.4,
                    "dataset": dataset,
                    "model": "CatBoost",
                    "model_type": "Foundation",
                    "preprocessing": "raw_nan",
                    "accuracy": m.get("accuracy"),
                    "f1": m.get("f1"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "roc_auc": m.get("roc_auc"),
                    "training_time_seconds": r.get("training_time_seconds"),
                    "missing_mechanism": None,
                    "missing_rate": None,
                })

    return rows


# ── Robustness analysis ───────────────────────────────────────────────────────

def _robustness_summary(df_43: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if df_43.empty:
        logger.warning("No Phase 4.3 data for robustness analysis")
        return pd.DataFrame()
    rob = (
        df_43.groupby(["model", "missing_mechanism"])
        .agg({"accuracy": ["mean", "std", "min", "max"], "f1": ["mean", "std"]})
        .round(4)
    )
    logger.info(f"Robustness summary:\n{rob}")
    return rob


# ── Visualizations ────────────────────────────────────────────────────────────

def _generate_visualizations(df_all: pd.DataFrame, df_43: pd.DataFrame,
                              logger: logging.Logger) -> None:
    if not VISUALIZATION_AVAILABLE:
        logger.warning("matplotlib/seaborn not available — skipping visualizations")
        return

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    if PLOT_STYLE in plt.style.available:
        plt.style.use(PLOT_STYLE)

    # 1. Missing rate vs performance — one plot per mechanism (MCAR / MAR / MNAR)
    if not df_43.empty:
        logger.info("Creating missing-rate comparison plots...")
        for mechanism in MISSINGNESS_PATTERNS:
            df_mech = df_43[df_43["missing_mechanism"] == mechanism]
            if df_mech.empty:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            fig.suptitle(
                f"Model Performance vs. Missing Rate — {mechanism}",
                fontsize=14, fontweight="bold",
            )

            for idx, metric in enumerate(["accuracy", "f1", "roc_auc"]):
                ax = axes[idx]
                for model in sorted(df_mech["model"].unique()):
                    df_m = df_mech[df_mech["model"] == model]
                    model_type = df_m["model_type"].iloc[0]
                    grouped = (
                        df_m.groupby("missing_rate")[metric]
                        .mean()
                        .sort_index()
                    )
                    linestyle = "-" if model_type == "Foundation" else "--"
                    linewidth = 2.5 if model_type == "Foundation" else 1.5
                    ax.plot(
                        grouped.index, grouped.values,
                        marker="o", label=model,
                        linestyle=linestyle, linewidth=linewidth,
                    )
                ax.set_xlabel("Missing Rate (%)", fontsize=11)
                ax.set_ylabel(metric.upper(), fontsize=11)
                ax.set_title(metric.upper(), fontsize=11)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(bottom=0.7)

            plt.tight_layout()
            out = VIZ_DIR / f"phase4_5_missing_rate_{mechanism}.png"
            plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
            logger.info(f"Saved {out.name}")
            plt.close()

    # 2. Classical vs Foundation boxplot comparison
    if not df_all.empty:
        logger.info("Creating classical vs. foundation comparison plot...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            "Classical vs. Foundation Models — Performance Distribution",
            fontsize=14, fontweight="bold",
        )
        for idx, metric in enumerate(["accuracy", "f1", "roc_auc"]):
            ax = axes[idx]
            data = df_all.dropna(subset=[metric])
            if not data.empty:
                sns.boxplot(
                    data=data, x="model_type", y=metric,
                    ax=ax, palette={"Classical": "#5B9BD5", "Foundation": "#ED7D31"},
                )
                ax.set_title(metric.upper(), fontsize=12, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel(metric.upper(), fontsize=10)
                ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        out = VIZ_DIR / "phase4_5_model_comparison.png"
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        logger.info(f"Saved {out.name}")
        plt.close()

    # 3. Stability heatmap (Phase 4.3 data — XGBoost / LightGBM across rates)
    if not df_43.empty:
        logger.info("Creating stability heatmap...")
        pivot = df_43.pivot_table(
            values="accuracy",
            index="model",
            columns="missing_rate",
            aggfunc="mean",
        ).round(3)

        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.7 + 1)))
            sns.heatmap(
                pivot, annot=True, fmt=".3f",
                cmap="RdYlGn", ax=ax,
                vmin=0.7, vmax=1.0,
                cbar_kws={"label": "Accuracy"},
            )
            ax.set_title(
                "Model Stability — Accuracy across Missing Rates (Phase 4.3)",
                fontsize=13, fontweight="bold",
            )
            ax.set_xlabel("Missing Rate (%)", fontsize=11)
            ax.set_ylabel("Model", fontsize=11)
            plt.tight_layout()
            out = VIZ_DIR / "phase4_5_stability_heatmap.png"
            plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
            logger.info(f"Saved {out.name}")
            plt.close()

    # 4. Per-dataset bar chart: all models, best preprocessing, accuracy
    if not df_all.empty:
        logger.info("Creating per-dataset model ranking chart...")
        best = (
            df_all[df_all["missing_rate"].isna()]  # baseline results (no missingness)
            .groupby(["dataset", "model", "model_type"], as_index=False)["accuracy"]
            .mean()
        )
        if not best.empty:
            datasets = best["dataset"].unique()
            fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=False)
            if len(datasets) == 1:
                axes = [axes]
            palette = {"Classical": "#5B9BD5", "Foundation": "#ED7D31"}
            for ax, ds in zip(axes, datasets):
                sub = best[best["dataset"] == ds].sort_values("accuracy", ascending=True)
                colors = [palette.get(t, "gray") for t in sub["model_type"]]
                ax.barh(sub["model"], sub["accuracy"], color=colors)
                ax.set_title(ds.replace("_", " ").title(), fontsize=11, fontweight="bold")
                ax.set_xlabel("Accuracy", fontsize=10)
                ax.set_xlim(left=0.85)
                ax.grid(True, alpha=0.3, axis="x")
                # legend
                from matplotlib.patches import Patch
                ax.legend(
                    handles=[Patch(color="#5B9BD5", label="Classical"),
                             Patch(color="#ED7D31", label="Foundation")],
                    fontsize=8,
                )
            plt.tight_layout()
            out = VIZ_DIR / "phase4_5_per_dataset_ranking.png"
            plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
            logger.info(f"Saved {out.name}")
            plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger = _setup_logging("phase4_5_analysis")
    logger.info("=" * 80)
    logger.info("PHASE 4.5: Final Analysis and Visualization")
    logger.info("=" * 80)

    # Load all results
    r41 = _load_json(RESULTS_DIR / "phase4_experiment_results.json", logger)
    r42 = _load_json(RESULTS_DIR / "phase4_2_experiment_results.json", logger)
    r43 = _load_json(RESULTS_DIR / "phase4_3_gradient_boosting_results.json", logger)
    r44 = _load_json(RESULTS_DIR / "phase4_4_tabicl_results.json", logger)

    # Parse into rows
    rows: list = []
    if r41:
        rows += _parse_41_42(r41, 4.1)
        logger.info(f"Phase 4.1: {len([r for r in rows if r['phase']==4.1])} rows")
    if r42:
        before = len(rows)
        rows += _parse_41_42(r42, 4.2)
        logger.info(f"Phase 4.2: {len(rows)-before} rows")
    if r43:
        before = len(rows)
        rows += _parse_43(r43)
        logger.info(f"Phase 4.3: {len(rows)-before} rows")
    if r44:
        before = len(rows)
        rows += _parse_44_tabicl(r44)
        logger.info(f"Phase 4.4 (TabICL + CatBoost): {len(rows)-before} rows")

    df_all = pd.DataFrame(rows)
    logger.info(f"Total consolidated rows: {len(df_all)}")
    logger.info(f"Models: {sorted(df_all['model'].unique())}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save consolidated
    df_all.to_csv(RESULTS_DIR / "phase4_5_consolidated_results.csv", index=False)
    logger.info("Saved phase4_5_consolidated_results.csv")

    # Save classical / foundation splits
    df_classical = df_all[df_all["model_type"] == "Classical"]
    df_foundation = df_all[df_all["model_type"] == "Foundation"]
    df_classical.to_csv(RESULTS_DIR / "phase4_5_classical_models.csv", index=False)
    df_foundation.to_csv(RESULTS_DIR / "phase4_5_foundation_models.csv", index=False)
    logger.info(f"Classical rows: {len(df_classical)}, Foundation rows: {len(df_foundation)}")

    # Phase 4.3 subset for robustness
    df_43 = df_all[df_all["phase"] == 4.3].copy()

    # Robustness analysis
    rob = _robustness_summary(df_43, logger)
    if not rob.empty:
        rob.to_csv(RESULTS_DIR / "phase4_5_robustness_analysis.csv")
        logger.info("Saved phase4_5_robustness_analysis.csv")

    # Visualizations
    _generate_visualizations(df_all, df_43, logger)

    print("\n✓ Phase 4.5 complete.")
    print(f"  Consolidated: {RESULTS_DIR / 'phase4_5_consolidated_results.csv'}")
    print(f"  Foundation models: {len(df_foundation)} rows ({sorted(df_foundation['model'].unique())})")
    print(f"  Visualizations: {VIZ_DIR}")


if __name__ == "__main__":
    main()