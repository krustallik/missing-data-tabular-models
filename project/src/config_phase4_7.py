"""Phase 4.7 configuration - Practical Usability Evaluation.

Phase 4.7 evaluates practical deployment suitability of models for real-world
classification tasks with incomplete data.

Focus areas:
1. Deployment complexity (installation, data prep, time, resources)
2. Practical suitability across missing rates (5%-40%)
3. Foundation vs classical model trade-offs
4. Production readiness assessment
5. Real-world deployment recommendations
"""

from config import RANDOM_STATE, TEST_SIZE

# Phase 4.7 Configuration
EVALUATE_DEPLOYMENT_COMPLEXITY = True
EVALUATE_PRACTICAL_SUITABILITY = True
GENERATE_DEPLOYMENT_GUIDE = True
GENERATE_PRACTICAL_RECOMMENDATIONS = True

# Model evaluation settings
INCLUDE_CLASSICAL_MODELS = True
INCLUDE_GRADIENT_BOOSTING = True
INCLUDE_FOUNDATION_MODELS = True

# Scenarios to evaluate
SCENARIOS = [
    {
        "name": "Small Data (5K rows) + 5% missing",
        "rows": 5000,
        "missing_rate": 0.05,
    },
    {
        "name": "Medium Data (50K rows) + 15% missing",
        "rows": 50000,
        "missing_rate": 0.15,
    },
    {
        "name": "Large Data (500K rows) + 25% missing",
        "rows": 500000,
        "missing_rate": 0.25,
    },
    {
        "name": "Extreme (>1M rows) + Unknown missing",
        "rows": 1000000,
        "missing_rate": None,
    },
]

# Practical thresholds
PRACTICAL_ACCURACY_THRESHOLD = 0.90  # Minimum acceptable accuracy (90%)
ACCEPTABLE_TRAINING_TIME_SECONDS = 3600  # 1 hour max for training
ACCEPTABLE_MEMORY_MB = 2048  # 2GB RAM max recommended

# Report settings
USE_MARKDOWN = True
INCLUDE_TABLES = True
INCLUDE_DECISION_MATRIX = True
INCLUDE_SCENARIO_ANALYSIS = True

# Reuse global constants
RANDOM_STATE = RANDOM_STATE
TEST_SIZE = TEST_SIZE

# Key metrics to evaluate
METRICS = {
    "accuracy": "Primary metric (%))",
    "f1": "Secondary metric",
    "roc_auc": "Tertiary metric",
    "training_time_seconds": "Deployment complexity proxy",
    "inference_time_ms": "Production latency",
}
