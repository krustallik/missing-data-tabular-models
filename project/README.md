# Tabular Missing-Data Benchmark (Phase 3.1)

This project sets up a reproducible data preparation layer for tabular **classification** datasets.

Datasets:
- Taiwanese Bankruptcy Prediction (UCI id=572) - downloaded by code
- Polish Companies Bankruptcy (UCI id=365) - downloaded by code
- Slovak financial ratios dataset - added manually

## Project structure

- `data/raw/` - original datasets
- `data/processed/` - reserved for later preprocessing
- `data/splits/` - reproducible train/test splits
- `results/tables/` - generated overview/setup/templates
- `results/logs/` - logs
- `src/` - source code for setup pipeline
- `notebooks/` - exploratory notebooks

## Quick start

```bash
pip install -r requirements.txt
python src/run_dataset_setup.py
python src/rung_generate_missingness.py
```

## Notes

- This phase only prepares datasets and split artifacts.
- No missingness simulation (MCAR/MAR) is implemented yet.
- No model training is implemented yet.

