"""Measure class balance in processed datasets and their train/test splits."""
from pathlib import Path
import pandas as pd

ROOT = Path("project/data")
TARGET = "target"

def report(path: Path) -> None:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        print(f"  {path.name}: NO '{TARGET}' column (cols={list(df.columns)[:5]}...)")
        return
    n = len(df)
    vc = df[TARGET].value_counts().sort_index()
    pct = (vc / n * 100).round(3)
    minority = int(vc.min())
    val_estimate = max(1, int(minority * 0.2))
    print(f"  {path.name:45s}  n={n:5d}  classes={dict(vc)}  minority%={pct.min():.2f}%  "
          f"val20%~{val_estimate} minority")

print("=== processed/ (full datasets) ===")
for p in sorted((ROOT / "processed").glob("*.csv")):
    report(p)

print("\n=== splits/ (train/test) ===")
for p in sorted((ROOT / "splits").glob("*.csv")):
    report(p)
