"""Phase 4.6 configuration - Documentation and Report Generation.

Phase 4.6 consolidates findings from Phases 4.1-4.5 into a structured research report
with Results, Discussion, and Conclusion sections suitable for academic presentation.

This phase serves as the documentation basis for Student 2's research on missing data
handling in tabular classification, specifically focusing on:
- Missing data mechanisms (MCAR, MAR, MNAR)
- Imputation strategies (median, MICE, CatBoost native)
- Classical vs. Foundation model robustness
- Performance characteristics across missingness patterns
"""

from config import RANDOM_STATE, TEST_SIZE

# Report Configuration
REPORT_TITLE = "Impact of Missing Data Handling on Tabular Classification Models"
STUDENT_NAME = "Student 2"
INSTITUTION = "University"
DATE = "April 11, 2026"

# Report Sections to Generate
GENERATE_ABSTRACT = True
GENERATE_RESULTS = True
GENERATE_DISCUSSION = True
GENERATE_CONCLUSION = True
GENERATE_INTERPRETATION_POINTS = True

# Data sections to include
INCLUDE_CLASSICAL_MODELS = True
INCLUDE_ROBUSTNESS_ANALYSIS = True
INCLUDE_FOUNDATION_MODELS = True
INCLUDE_COMPARATIVE_ANALYSIS = True

# Report format options
USE_MARKDOWN = True           # Generate .md report
USE_LATEX = False             # Generate .tex (for LaTeX compilation)
USE_PDF = False               # Attempt PDF generation

# Table configuration
TABLE_FORMAT = "markdown"     # "markdown", "csv", "latex", "html"
INCLUDE_STATISTICAL_TESTS = True

# Reuse global constants
RANDOM_STATE = RANDOM_STATE
TEST_SIZE = TEST_SIZE

# Key metrics to highlight
KEY_METRICS = ["accuracy", "f1", "roc_auc"]

# Interpretation thresholds
PERFORMANCE_EXCELLENT = 0.95
PERFORMANCE_GOOD = 0.90
PERFORMANCE_ACCEPTABLE = 0.85
