#!/usr/bin/env python3
"""Check availability of foundation models."""

import sys
print(f"Python: {sys.version}\n")
print("Testing foundation model availability...\n")

models = {
    'CatBoost': 'catboost',
    'TabPFN': 'tabpfn',
    'TabICL': 'tabicl',
    'XGBoost': 'xgboost',
    'LightGBM': 'lightgbm'
}

for name, module in models.items():
    try:
        __import__(module)
        print(f"✓ {name:15} AVAILABLE")
    except ImportError:
        print(f"✗ {name:15} NOT AVAILABLE")
