"""Regenerate the documentation dataset chart from the bundled processed CSVs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "project" / "data" / "processed"
OUTPUT = ROOT / "docs" / "images" / "dataset-overview.png"
DATASETS = [
    ("polish_1year.csv", "Polish companies · 1-year"),
    ("slovak_manufacture_13.csv", "Slovak manufacturing · 13-year subset"),
    ("taiwan_bankruptcy.csv", "Taiwan bankruptcy"),
]


def dataset_summary(filename: str, label: str) -> dict:
    frame = pd.read_csv(DATA_DIR / filename)
    target = frame["target"]
    features = frame.drop(columns="target")
    unnamed = [column for column in features.columns if str(column).lower().startswith("unnamed:")]
    features = features.drop(columns=unnamed)
    counts = target.value_counts().to_dict()
    n_rows = len(frame)
    cells = features.shape[0] * features.shape[1]
    return {
        "label": label,
        "rows": n_rows,
        "features": features.shape[1],
        "nonbankrupt": int(counts.get(0, 0)),
        "bankrupt": int(counts.get(1, 0)),
        "bankrupt_pct": 100 * int(counts.get(1, 0)) / n_rows,
        "missing_pct": 100 * int(features.isna().sum().sum()) / cells if cells else 0,
    }


def main() -> None:
    rows = [dataset_summary(filename, label) for filename, label in DATASETS]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(13.5, 7.2), facecolor="#F4F7FA")
    fig.text(0.075, 0.925, "Dataset snapshot", fontsize=24, fontweight="bold", color="#14263D")
    fig.text(0.075, 0.875, "Three binary bankruptcy datasets included in the repository", fontsize=12, color="#536477")
    fig.legend(
        handles=[
            Patch(facecolor="#255B82", label="Class 0 · non-bankrupt"),
            Patch(facecolor="#E8894D", label="Class 1 · bankrupt"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.93, 0.905),
        frameon=False,
        fontsize=9,
    )

    for item, y in zip(rows, [0.68, 0.45, 0.22]):
        fig.text(0.075, y + 0.033, item["label"], fontsize=13, fontweight="bold", color="#14263D", va="center")
        fig.text(0.075, y - 0.012, f"{item['bankrupt']:,} bankrupt · {item['bankrupt_pct']:.2f}% of rows", fontsize=10, color="#6E4D36", va="center")

        ax = fig.add_axes([0.37, y - 0.052, 0.34, 0.105], facecolor="none")
        ax.barh(0, 100 - item["bankrupt_pct"], height=0.42, color="#255B82", edgecolor="none")
        ax.barh(0, item["bankrupt_pct"], left=100 - item["bankrupt_pct"], height=0.42, color="#E8894D", edgecolor="none")
        ax.text((100 - item["bankrupt_pct"]) / 2, 0, f"{item['nonbankrupt']:,} non-bankrupt", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis="x", labelsize=8, colors="#748397", length=0, pad=5)
        ax.grid(axis="x", color="#DCE4EC", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

        card = FancyBboxPatch((0.745, y - 0.058), 0.205, 0.118, boxstyle="round,pad=0.012,rounding_size=0.012", transform=fig.transFigure, facecolor="white", edgecolor="#E1E8EF", linewidth=0.8)
        fig.add_artist(card)
        for x, heading, value in [
            (0.765, "ROWS", f"{item['rows']:,}"),
            (0.835, "FEATURES", str(item["features"])),
            (0.895, "NATIVE NA", f"{item['missing_pct']:.2f}%"),
        ]:
            fig.text(x, y + 0.017, heading, fontsize=7.5, fontweight="bold", color="#8795A6")
            fig.text(x, y - 0.023, value, fontsize=12, fontweight="bold", color="#14263D")

    fig.text(0.075, 0.065, "Class proportions and missingness are calculated from project/data/processed/*.csv. Unnamed index-artifact columns are excluded, matching the model data loader.", fontsize=8.5, color="#6E7D8F")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
