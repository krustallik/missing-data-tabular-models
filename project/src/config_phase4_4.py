"""Phase 4.4 configuration - CatBoost (Foundation Model for Tabular Data).

CatBoost is a gradient boosting library designed for tabular classification tasks.
Phase 4.4 tests CatBoost performance with different imputation strategies and raw NaN handling.

Note: CatBoost can handle categorical features natively and has some NaN tolerance.
"""

from config import RANDOM_STATE, TEST_SIZE

# CatBoost test configurations
TEST_WITH_IMPUTATION = True
TEST_WITHOUT_IMPUTATION = True

# Use same preprocessing as Phase 4.1/4.2
PREPROCESSING_METHODS = ["median", "mice"]

# Reuse from global config
RANDOM_STATE = RANDOM_STATE
TEST_SIZE = TEST_SIZE

# CatBoost specific parameters
try:
    import catboost  # noqa: F401
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
