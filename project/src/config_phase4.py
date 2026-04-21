"""Phase 4.1 - Enhanced configuration for model training and evaluation."""

# Existing Phase 3.1 configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
METRICS = ["accuracy", "f1", "roc_auc", "precision", "recall"]
RESULT_COLUMNS = [
    "dataset",
    "missing_type",
    "missing_rate",
    "preprocessing",
    "model",
    "accuracy",
    "f1",
    "roc_auc",
    "precision",
    "recall",
]

# Phase 4.1 - Model Training Configuration
CV_FOLDS = 5
SCALING_ENABLED = True

# Preprocessing / missing-data handling options evaluated in Phase 4.1
# - median: median imputation + scaling
# - mice: IterativeImputer (MICE) + scaling
# - mice_indicator: MICE + add binary missingness indicators per feature
PREPROCESSING_METHODS = [
    "median",
    "mice",
    "mice_indicator",
]

# Logistic Regression Hyperparameters
LR_PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10],
    "penalty": ["l2"],
}

# Random Forest Hyperparameters
RF_PARAM_GRID = {
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "n_estimators": [50, 100, 200],
}

# Experiment settings
EXPERIMENT_TIMEOUT_SECONDS = 3600  # 1 hour per dataset
VERBOSE_LEVEL = 1  # 0=silent, 1=normal, 2=verbose
