"""Phase 4.5 configuration - Final Analysis and Visualization.

Phase 4.5 consolidates results from all experimental phases (4.1-4.4) and generates
comprehensive visualizations comparing:
- Classical models vs. Foundation models
- Different missingness patterns (MCAR, MAR, MNAR)
- Model stability across missing rates
- Preprocessing strategy effectiveness
"""

from config import RANDOM_STATE, TEST_SIZE

# Analysis configuration
ANALYSIS_METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]

# Visualization configuration
PLOT_DPI = 150
PLOT_STYLE = "seaborn-v0_8-darkgrid"

# Model grouping for comparisons
CLASSICAL_MODELS = ["LogisticRegression", "RandomForest", "SVM", "MLP"]
FOUNDATION_MODELS = ["CatBoost"]

# Missingness patterns for Phase 4.3 analysis
MISSINGNESS_PATTERNS = ["MCAR", "MAR", "MNAR"]
MISSING_RATES = [5, 10, 15, 20, 30, 40]

# Preprocessing methods used across phases
PREPROCESSING_METHODS = ["median", "mice", "mice_indicator"]

# Output configuration
CONSOLIDATION_ENABLED = True
VISUALIZATION_ENABLED = True
STABILITY_ANALYSIS_ENABLED = True

# Reuse global constants
RANDOM_STATE = RANDOM_STATE
TEST_SIZE = TEST_SIZE
