"""Phase 4.3 configuration - Gradient Boosting robustness on missingness scenarios.

Student 2 scenario: Test Gradient Boosting (XGBoost, LightGBM) sensitivity to
missing values at varying rates and mechanisms (MCAR/MAR/MNAR).

If Gradient Boosting models are unavailable, falls back to SVM/MLP.
"""

from config import RANDOM_STATE, TEST_SIZE

# Gradient Boosting models (Student 2 focus)
# Falls back to SVM/MLP if XGBoost/LightGBM not available
GRADIENT_BOOSTING_MODELS = ["gradient_boosting", "xgboost", "lightgbm"]
FALLBACK_MODELS = ["svm", "mlp"]

# Missing value rates to inject
MISSING_RATES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

# Missing mechanisms
MISSING_MECHANISMS = ["MCAR", "MAR", "MNAR"]

# Preprocessing methods
PREPROCESSING_METHODS = ["median", "mice"]

# Cross-validation folds
CV_FOLDS = 5

# Reuse from global config
RANDOM_STATE = RANDOM_STATE
TEST_SIZE = TEST_SIZE
