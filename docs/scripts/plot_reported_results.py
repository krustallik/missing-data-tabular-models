"""Regenerate charts from numeric tables in the co-authored report.

These are report-level results transcribed into this script. They are not
calculated from the current checkout and do not imply experiment reproduction.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["Polish 1-year", "Slovak Manufacture 13", "Taiwan"]
FOUNDATION = [0.6483, 0.2760, 0.4740]
CLASSICAL = [0.4030, 0.1795, 0.4185]

IMPUTATION_STRATEGIES = [
    "No explicit imputation",
    "MICE + missingness indicator",
    "Mean",
    "MICE",
    "Median",
    "kNN",
]
IMPUTATION_VALUES = np.array(
    [
        [0.6247, 0.2645, 0.4754],
        [0.5396, 0.2045, 0.4242],
        [0.4524, 0.1906, 0.4263],
        [0.4516, 0.1908, 0.4304],
        [0.4035, 0.1896, 0.4261],
        [0.3528, 0.1953, 0.4235],
    ]
)

MISSINGNESS_ENDPOINTS = {
    "Polish 1-year": {
        "TabPFN": (0.8923, 0.4275),
        "TabICL": (0.8339, 0.2318),
        "LightGBM": (0.7171, 0.3895),
        "XGBoost": (0.7168, 0.3872),
    },
    "Taiwan": {
        "XGBoost": (0.5194, 0.4640),
        "LightGBM": (0.5191, 0.4644),
        "TabICL": (0.5064, 0.4493),
        "TabPFN": (0.4942, 0.4363),
    },
}
SLOVAK_TABICL_MECHANISMS = {"MCAR": 0.3087, "MAR": 0.3041, "MNAR": 0.2998}

COLORS = {
    "foundation": "#A9531E",
    "classical": "#46677F",
    "TabPFN": "#A9531E",
    "TabICL": "#46677F",
    "LightGBM": "#879451",
    "XGBoost": "#7C659B",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_group_pr_auc() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(DATASETS))
    width = 0.34
    bars_foundation = ax.bar(
        x - width / 2,
        FOUNDATION,
        width,
        label="Foundation models (TabPFN, TabICL)",
        color=COLORS["foundation"],
    )
    bars_classical = ax.bar(
        x + width / 2,
        CLASSICAL,
        width,
        label="Classical models",
        color=COLORS["classical"],
    )
    ax.bar_label(bars_foundation, fmt="%.4f", padding=3, fontsize=9)
    ax.bar_label(bars_classical, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xticks(x, DATASETS)
    ax.set_ylim(0, 0.78)
    ax.set_ylabel("Reported group-average PR-AUC")
    ax.set_title("Historical group comparison from the co-authored report", loc="left", pad=16)
    ax.grid(axis="y", color="#E7E2DA", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    fig.text(
        0.01,
        -0.015,
        "Report-level averages across its tested scenarios; not reproduced from this repository checkout.",
        color="#5E5A54",
        fontsize=9,
    )
    save(fig, "reported-group-pr-auc.png")


def plot_imputation_pr_auc() -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    image = ax.imshow(IMPUTATION_VALUES, cmap="YlOrBr", vmin=0, vmax=0.70, aspect="auto")
    ax.set_xticks(np.arange(3), DATASETS)
    ax.set_yticks(np.arange(len(IMPUTATION_STRATEGIES)), IMPUTATION_STRATEGIES)
    ax.set_title("Reported average PR-AUC by missing-data strategy", loc="left", pad=16)
    for row in range(IMPUTATION_VALUES.shape[0]):
        for col in range(IMPUTATION_VALUES.shape[1]):
            value = IMPUTATION_VALUES[row, col]
            ax.text(
                col,
                row,
                f"{value:.4f}",
                ha="center",
                va="center",
                color="white" if value >= 0.47 else "#30271E",
                fontsize=9,
                fontweight="bold" if row == 0 else "normal",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.035)
    colorbar.set_label("Reported PR-AUC")
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.01,
        -0.015,
        "Averages over the report's models, missingness mechanisms, and rates; not current-checkout results.",
        color="#5E5A54",
        fontsize=9,
    )
    save(fig, "reported-imputation-pr-auc.png")


def plot_missingness_robustness() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})
    for ax, dataset in ((axes[0], "Polish 1-year"), (axes[2], "Taiwan")):
        for model, (at_5, at_40) in MISSINGNESS_ENDPOINTS[dataset].items():
            ax.plot(
                [5, 40],
                [at_5, at_40],
                marker="o",
                markersize=5,
                linewidth=2,
                color=COLORS[model],
                label=model,
            )
        ax.set_xticks([5, 40], ["5%", "40%"])
        ax.set_xlim(2, 43)
        ax.set_ylim(0, 1.0 if dataset.startswith("Polish") else 0.60)
        ax.set_ylabel("Reported PR-AUC")
        ax.set_title(dataset, loc="left")
        ax.grid(axis="y", color="#E7E2DA", linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, loc="best")

    ax = axes[1]
    mechanisms = list(SLOVAK_TABICL_MECHANISMS)
    scores = list(SLOVAK_TABICL_MECHANISMS.values())
    ax.plot(mechanisms, scores, marker="o", color=COLORS["TabICL"], linewidth=2)
    for x, y in enumerate(scores):
        ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_ylim(0.285, 0.32)
    ax.set_ylabel("Reported PR-AUC")
    ax.set_title("Slovak · TabICL", loc="left")
    ax.grid(axis="y", color="#E7E2DA", linewidth=0.8)
    ax.text(0.02, 0.02, "By mechanism, not rate", transform=ax.transAxes, fontsize=8, color="#5E5A54")

    fig.suptitle("Selected historical results under missingness", x=0.01, ha="left", y=1.03)
    fig.text(
        0.01,
        -0.03,
        "Polish and Taiwan show report-stated 5%/40% endpoints; connecting lines do not imply measured intermediate values. Slovak values compare MCAR/MAR/MNAR.",
        color="#5E5A54",
        fontsize=8.5,
    )
    save(fig, "reported-missingness-robustness.png")


if __name__ == "__main__":
    plot_group_pr_auc()
    plot_imputation_pr_auc()
    plot_missingness_robustness()
