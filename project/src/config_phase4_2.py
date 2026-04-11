"""Phase 4.2 configuration for extended models."""

# Extended model hyperparameter grids
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "poly"],
    "gamma": ["scale", "auto"],
}

MLP_PARAM_GRID = {
    "hidden_layer_sizes": [(50,), (100, 50), (100, 50, 25)],
    "alpha": [0.0001, 0.001],
    "learning_rate": ["constant", "adaptive"],
}

XGBOOST_PARAM_GRID = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.7, 0.9],
}

LIGHTGBM_PARAM_GRID = {
    "max_depth": [5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63],
}

# Training settings
CV_FOLDS = 5
SCALING_ENABLED = True

# Preprocessing / missing-data handling options evaluated in Phase 4.2
# REQUIRED: mice_indicator must be included for Student 2's missingness analysis
# - median: Simple median imputation + scaling
# - mice: IterativeImputer (MICE algorithm) + scaling  
# - mice_indicator: MICE + binary missing indicator features (REQUIRED for sensitivity analysis)
PREPROCESSING_METHODS = [
    "median",
    "mice",
    "mice_indicator",  # REQUIRED: Missing indicator strategy for robustness testing
]
