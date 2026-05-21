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

from config import (
    METRICS,
    OUTPUT_FILES,
    PRIMARY_METRICS,
    VIZ_DIR,
    ensure_output_dirs,
    ranking_metric_column,
)
from data_utils import setup_logging
from models import model_type as _resolve_model_type


# Metric shown on heatmaps / per-dataset ranking (see config.RANKING_METRIC).
def _ranking_metric(df: pd.DataFrame) -> str:
    return ranking_metric_column(df)


def _metrics_in_df(df: pd.DataFrame) -> list:
    """Return the subset of PRIMARY_METRICS that actually exist in ``df``."""
    return [m for m in PRIMARY_METRICS if m in df.columns]


try:
    import matplotlib

    # Non-interactive backend: saves PNG without Tcl/Tk (common Windows venv issue).
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    VIZ_AVAILABLE = True
except Exception:
    VIZ_AVAILABLE = False


PLOT_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"


NATIVE_MECHANISM = "native"
NATIVE_RATE = 0.0


def _normalize_native(df: pd.DataFrame) -> pd.DataFrame:
    """Convert legacy NaN rows for the no-injection scenario to the stable
    ``("native", 0.0)`` representation now written by ``experiment_runner``.

    This keeps consolidation/reporting working uniformly regardless of which
    schema the CSV on disk was written with.
    """
    if df.empty or "missing_mechanism" not in df.columns:
        return df
    df = df.copy()
    df["missing_mechanism"] = df["missing_mechanism"].where(
        df["missing_mechanism"].notna(), NATIVE_MECHANISM,
    )
    if "missing_rate" in df.columns:
        native_mask = df["missing_mechanism"] == NATIVE_MECHANISM
        df.loc[native_mask & df["missing_rate"].isna(), "missing_rate"] = NATIVE_RATE
    return df


def _load_experiment_results(logger: logging.Logger) -> Optional[pd.DataFrame]:
    path = OUTPUT_FILES["experiment_results"]
    if not path.exists():
        logger.error(f"Experiment results not found: {path}")
        return None
    df = pd.read_csv(path)
    df = _normalize_native(df)
    if "model" in df.columns:
        df["model_type"] = df["model"].map(_resolve_model_type)
    logger.info(f"Loaded {path.name}: {len(df)} rows")
    return df


def _aggregate_across_seeds(
    df: pd.DataFrame, logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Mean metrics over split random states and experiment seeds."""
    n_seeds = int(df["seed"].nunique(dropna=True)) if "seed" in df.columns else 1
    n_split_seeds = int(df["split_seed"].nunique(dropna=True)) if "split_seed" in df.columns else 1
    if n_seeds <= 1 and n_split_seeds <= 1:
        return df

    group_cols = [
        "dataset", "missing_mechanism", "missing_rate", "imputation",
        "model", "model_type",
    ]
    present = [c for c in group_cols if c in df.columns]
    metric_cols = [c for c in METRICS if c in df.columns]
    extra = [c for c in ("threshold", "training_time_seconds") if c in df.columns]

    agg: dict = {c: "mean" for c in metric_cols + extra}
    out = df.groupby(present, as_index=False, dropna=False).agg(agg)
    if logger is not None:
        logger.info(
            f"Aggregated {len(df)} repeated rows -> {len(out)} rows "
            f"(mean over {n_split_seeds} split seeds x {n_seeds} experiment seeds)"
        )
    return out


def _save_consolidated(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df_plot = _aggregate_across_seeds(df, logger=logger)
    out = OUTPUT_FILES["consolidated_results"]
    df_plot.to_csv(out, index=False)
    logger.info(f"Saved {out.name}: {len(df_plot)} rows")

    classical = df_plot[df_plot["model_type"] == "Classical"]
    foundation = df_plot[df_plot["model_type"] == "Foundation"]
    classical.to_csv(OUTPUT_FILES["classical_models"], index=False)
    foundation.to_csv(OUTPUT_FILES["foundation_models"], index=False)
    logger.info(
        f"Saved classical_models.csv ({len(classical)} rows) and "
        f"foundation_models.csv ({len(foundation)} rows)"
    )
    return df_plot


def _robustness(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    # Robustness summary is per missingness mechanism, so the native (no
    # injection) rows must be excluded. Accept both the new "native" sentinel
    # and legacy NaN rows.
    sub = df.dropna(subset=["accuracy", "missing_mechanism"])
    sub = sub[sub["missing_mechanism"] != NATIVE_MECHANISM]
    if sub.empty:
        logger.warning("No missingness-annotated rows for robustness analysis")
        return pd.DataFrame()

    agg_spec = {
        "accuracy_mean": ("accuracy", "mean"),
        "accuracy_std": ("accuracy", "std"),
        "accuracy_min": ("accuracy", "min"),
        "accuracy_max": ("accuracy", "max"),
        "f1_mean": ("f1", "mean"),
        "f1_std": ("f1", "std"),
        "n": ("accuracy", "size"),
    }
    for m in ("pr_auc", "balanced_accuracy", "f1_macro", "recall_class1"):
        if m in sub.columns:
            agg_spec[f"{m}_mean"] = (m, "mean")
            agg_spec[f"{m}_std"] = (m, "std")

    grp = sub.groupby(["model", "missing_mechanism"]).agg(**agg_spec).round(4)
    out = OUTPUT_FILES["robustness_analysis"]
    grp.to_csv(out)
    logger.info(f"Saved {out.name}: {len(grp)} rows (metrics: {list(agg_spec.keys())})")
    return grp


def _plot_missing_rate_curves(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE:
        return
    sub = df.dropna(subset=["missing_rate", "accuracy", "missing_mechanism"])
    # Exclude the native sentinel rows; the curves are about injected
    # missingness only.
    sub = sub[sub["missing_mechanism"] != NATIVE_MECHANISM]
    if sub.empty:
        logger.warning("No data for missing-rate curves")
        return
    if PLOT_STYLE in plt.style.available:
        plt.style.use(PLOT_STYLE)

    metrics = _metrics_in_df(sub)  # PRIMARY_METRICS that are actually populated
    if not metrics:
        metrics = ["accuracy"]
    n = len(metrics)

    for mech in sorted(sub["missing_mechanism"].unique()):
        df_mech = sub[sub["missing_mechanism"] == mech]
        if df_mech.empty:
            continue
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5), squeeze=False)
        axes = axes[0]
        fig.suptitle(f"Model Performance vs Missing Rate - {mech}",
                     fontsize=14, fontweight="bold")
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            for model in sorted(df_mech["model"].dropna().unique()):
                df_m = df_mech[df_mech["model"] == model].dropna(subset=[metric])
                if df_m.empty:
                    continue
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
            ax.set_ylabel(metric)
            ax.set_title(metric.replace("_", " "))
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = VIZ_DIR / f"missing_rate_{mech}.png"
        plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {out.name} (metrics={metrics})")


def _plot_classical_vs_foundation(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE or df.empty:
        return
    data = df.dropna(subset=["accuracy"])
    if data.empty:
        return
    metrics = _metrics_in_df(data)
    if not metrics:
        metrics = ["accuracy"]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]
    fig.suptitle("Classical vs Foundation Models - Performance Distribution",
                 fontsize=14, fontweight="bold")
    for idx, metric in enumerate(metrics):
        sub = data.dropna(subset=[metric])
        if sub.empty:
            continue
        sns.boxplot(
            data=sub, x="model_type", y=metric, ax=axes[idx],
            palette={"Classical": "#5B9BD5", "Foundation": "#ED7D31"},
        )
        axes[idx].set_title(metric.replace("_", " "), fontweight="bold")
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel(metric)
        axes[idx].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = VIZ_DIR / "classical_vs_foundation.png"
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out.name} (metrics={metrics})")


def _plot_stability_heatmap(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE:
        return
    metric = _ranking_metric(df)
    sub = df.dropna(subset=["missing_rate", metric])
    if sub.empty:
        return
    pivot = sub.pivot_table(
        values=metric, index="model", columns="missing_rate", aggfunc="mean",
    ).round(3)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6 + 1)))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
        vmin=max(0.0, float(pivot.min().min())),
        vmax=1.0,
        cbar_kws={"label": metric},
    )
    ax.set_title(f"Model Stability - {metric} across Missing Rates",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Missing Rate (%)")
    ax.set_ylabel("Model")
    plt.tight_layout()
    out = VIZ_DIR / "stability_heatmap.png"
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out.name} (metric={metric})")


def _plot_per_dataset_ranking(df: pd.DataFrame, logger: logging.Logger) -> None:
    if not VIZ_AVAILABLE:
        return
    metric = _ranking_metric(df)
    # Per-dataset ranking is drawn on the native scenario. Accept both the
    # new "native" sentinel and legacy NaN rows.
    native_mask = (
        df["missing_mechanism"].isna()
        | (df["missing_mechanism"] == NATIVE_MECHANISM)
    )
    sub = df[native_mask].dropna(subset=[metric])
    if sub.empty:
        return
    best = sub.groupby(["dataset", "model", "model_type"], as_index=False)[metric].mean()
    if best.empty:
        return
    datasets = list(best["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    from matplotlib.patches import Patch

    palette = {"Classical": "#5B9BD5", "Foundation": "#ED7D31"}
    for ax, ds in zip(axes, datasets):
        s = best[best["dataset"] == ds].sort_values(metric, ascending=True)
        colors = [palette.get(t, "gray") for t in s["model_type"]]
        ax.barh(s["model"], s[metric], color=colors)
        ax.set_title(ds.replace("_", " ").title(), fontweight="bold")
        ax.set_xlabel(f"{metric} (native scenario)")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(left=max(0.0, float(s[metric].min() - 0.05)))
        ax.legend(handles=[
            Patch(color=palette["Classical"], label="Classical"),
            Patch(color=palette["Foundation"], label="Foundation"),
        ], fontsize=8)
    plt.tight_layout()
    out = VIZ_DIR / "per_dataset_ranking.png"
    plt.savefig(out, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out.name} (metric={metric})")


def consolidate(logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    ensure_output_dirs()
    if logger is None:
        logger = setup_logging("consolidation")

    df = _load_experiment_results(logger)
    if df is None or df.empty:
        logger.error("No experiment results to consolidate")
        return None

    df_agg = _save_consolidated(df, logger)
    _robustness(df_agg, logger)

    if not VIZ_AVAILABLE:
        logger.warning("matplotlib/seaborn not installed; skipping visualizations")
        return df_agg

    _plot_missing_rate_curves(df_agg, logger)
    _plot_classical_vs_foundation(df_agg, logger)
    _plot_stability_heatmap(df_agg, logger)
    _plot_per_dataset_ranking(df_agg, logger)
    return df_agg


if __name__ == "__main__":
    consolidate()
