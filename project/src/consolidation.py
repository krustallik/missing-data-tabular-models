"""Consolidation, robustness summary, and visualizations.

Reads ``experiment_results.csv`` and produces:

- ``consolidated_results.csv`` — same table, dropped ``error`` column for
  rows that succeeded; kept as-is (error column intact) for ease of analysis.
- ``classical_models.csv`` / ``foundation_models.csv`` — convenience splits.
- ``robustness_analysis.csv`` — per-model, per-mechanism aggregates.
- ``visualizations/*.png`` — the standard four plots (per-mechanism curves,
  classical-vs-foundation boxplot, stability heatmap, per-dataset ranking).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import OUTPUT_FILES, VIZ_DIR, ensure_output_dirs
from data_utils import setup_logging


try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    VIZ_AVAILABLE = True
except Exception:
    VIZ_AVAILABLE = False


PLOT_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"


def _load_experiment_results(logger: logging.Logger) -> Optional[pd.DataFrame]:
    path = OUTPUT_FILES["experiment_results"]
    if not path.exists():
        logger.error(f"Experiment results not found: {path}")
        return None
    df = pd.read_csv(path)
    logger.info(f"Loaded {path.name}: {len(df)} rows")
    return df


def _save_consolidated(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    out = OUTPUT_FILES["consolidated_results"]
    df.to_csv(out, index=False)
    logger.info(f"Saved {out.name}: {len(df)} rows")

    classical = df[df["model_type"] == "Classical"]
    foundation = df[df["model_type"] == "Foundation"]
    classical.to_csv(OUTPUT_FILES["classical_models"], index=False)
    foundation.to_csv(OUTPUT_FILES["foundation_models"], index=False)
    logger.info(
        f"Saved classical_models.csv ({len(classical)} rows) and "
        f"foundation_models.csv ({len(foundation)} rows)"
    )
    return df


def _robustness(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    sub = df.dropna(subset=["accuracy", "missing_mechanism"])
    if sub.empty:
        logger.warning("No missingness-annotated rows for robustness analysis")
        return pd.DataFrame()
    grp = sub.groupby(["model", "missing_mechanism"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        n=("accuracy", "size"),
    ).round(4)
    out = OUTPUT_FILES["robustness_analysis"]
    grp.to_csv(out)
    logger.info(f"Saved {out.name}: {len(grp)} rows")
    return grp


def _plot_missing_rate_curves(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE:
        return
    sub = df.dropna(subset=["missing_rate", "accuracy", "missing_mechanism"])
    if sub.empty:
        logger.warning("No data for missing-rate curves")
        return
    if PLOT_STYLE in plt.style.available:
        plt.style.use(PLOT_STYLE)

    for mech in sorted(sub["missing_mechanism"].unique()):
        df_mech = sub[sub["missing_mechanism"] == mech]
        if df_mech.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Model Performance vs Missing Rate - {mech}",
                     fontsize=14, fontweight="bold")
        for idx, metric in enumerate(["accuracy", "f1", "roc_auc"]):
            ax = axes[idx]
            for model in sorted(df_mech["model"].dropna().unique()):
                df_m = df_mech[df_mech["model"] == model]
                mtype = df_m["model_type"].iloc[0]
                grouped = df_m.groupby("missing_rate")[metric].mean().sort_index()
                if grouped.empty:
                    continue
                ax.plot(
                    grouped.index, grouped.values, marker="o", label=model,
                    linestyle="-" if mtype == "Foundation" else "--",
                    linewidth=2.5 if mtype == "Foundation" else 1.5,
                )
            ax.set_xlabel("Missing Rate (%)")
            ax.set_ylabel(metric.upper())
            ax.set_title(metric.upper())
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = VIZ_DIR / f"missing_rate_{mech}.png"
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {out.name}")


def _plot_classical_vs_foundation(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE or df.empty:
        return
    data = df.dropna(subset=["accuracy"])
    if data.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Classical vs Foundation Models - Performance Distribution",
                 fontsize=14, fontweight="bold")
    for idx, metric in enumerate(["accuracy", "f1", "roc_auc"]):
        sub = data.dropna(subset=[metric])
        if sub.empty:
            continue
        sns.boxplot(
            data=sub, x="model_type", y=metric, ax=axes[idx],
            palette={"Classical": "#5B9BD5", "Foundation": "#ED7D31"},
        )
        axes[idx].set_title(metric.upper(), fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel(metric.upper())
        axes[idx].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = VIZ_DIR / "classical_vs_foundation.png"
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out.name}")


def _plot_stability_heatmap(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE:
        return
    sub = df.dropna(subset=["missing_rate", "accuracy"])
    if sub.empty:
        return
    pivot = sub.pivot_table(
        values="accuracy", index="model", columns="missing_rate", aggfunc="mean",
    ).round(3)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6 + 1)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
        vmin=max(0.5, float(pivot.min().min())),
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
    )
    ax.set_title("Model Stability - Accuracy across Missing Rates",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Missing Rate (%)")
    ax.set_ylabel("Model")
    plt.tight_layout()
    out = VIZ_DIR / "stability_heatmap.png"
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out.name}")


def _plot_per_dataset_ranking(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE:
        return
    sub = df[df["missing_mechanism"].isna()].dropna(subset=["accuracy"])
    if sub.empty:
        return
    best = sub.groupby(["dataset", "model", "model_type"], as_index=False)["accuracy"].mean()
    if best.empty:
        return
    datasets = list(best["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    from matplotlib.patches import Patch

    palette = {"Classical": "#5B9BD5", "Foundation": "#ED7D31"}
    for ax, ds in zip(axes, datasets):
        s = best[best["dataset"] == ds].sort_values("accuracy", ascending=True)
        colors = [palette.get(t, "gray") for t in s["model_type"]]
        ax.barh(s["model"], s["accuracy"], color=colors)
        ax.set_title(ds.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel("Accuracy (native scenario)")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(left=max(0.0, float(s["accuracy"].min() - 0.05)))
        ax.legend(handles=[
            Patch(color=palette["Classical"], label="Classical"),
            Patch(color=palette["Foundation"], label="Foundation"),
        ], fontsize=8)
    plt.tight_layout()
    out = VIZ_DIR / "per_dataset_ranking.png"
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out.name}")


def consolidate(logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    ensure_output_dirs()
    if logger is None:
        logger = setup_logging("consolidation")

    df = _load_experiment_results(logger)
    if df is None or df.empty:
        logger.error("No experiment results to consolidate")
        return None

    df = _save_consolidated(df, logger)
    _robustness(df, logger)

    if not VIZ_AVAILABLE:
        logger.warning("matplotlib/seaborn not installed; skipping visualizations")
        return df

    _plot_missing_rate_curves(df, logger)
    _plot_classical_vs_foundation(df, logger)
    _plot_stability_heatmap(df, logger)
    _plot_per_dataset_ranking(df, logger)
    return df


if __name__ == "__main__":
    consolidate()
