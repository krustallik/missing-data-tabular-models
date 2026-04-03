"""Global configuration constants for reproducible dataset setup."""

RANDOM_STATE = 42
TEST_SIZE = 0.2
METRICS = ["accuracy", "f1", "roc_auc"]
RESULT_COLUMNS = [
    "dataset",
    "missing_type",
    "missing_rate",
    "preprocessing",
    "model",
    "accuracy",
    "f1",
    "roc_auc",
]

