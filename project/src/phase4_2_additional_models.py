"""Phase 4.2 - Extended Models: SVM, XGBoost, LightGBM, Neural Networks."""

from typing import Dict
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from config import RANDOM_STATE
from config_phase4_2 import PREPROCESSING_METHODS


class ExtendedModelEvaluator:
    """Extended model support for Phase 4.2."""
    
    MODELS = {
        "svm": SVC(kernel='rbf', random_state=RANDOM_STATE, probability=True),
        "mlp": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
    }
    
    if HAS_XGBOOST:
        MODELS["xgboost"] = xgb.XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0
        )
    
    if HAS_LIGHTGBM:
        MODELS["lightgbm"] = lgb.LGBMClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    
    PARAM_GRIDS = {
        "svm": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "poly"],
            "gamma": ["scale", "auto"],
        },
        "mlp": {
            "hidden_layer_sizes": [(50,), (100, 50), (100, 50, 25)],
            "alpha": [0.0001, 0.001],
            "learning_rate": ["constant", "adaptive"],
        },
    }
    
    if HAS_XGBOOST:
        PARAM_GRIDS["xgboost"] = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.7, 0.9],
        }
    
    if HAS_LIGHTGBM:
        PARAM_GRIDS["lightgbm"] = {
            "max_depth": [5, 7, 10],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [31, 63],
        }
    
    def __init__(self, model_name: str, preprocessing: str, logger: logging.Logger):
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        if preprocessing not in PREPROCESSING_METHODS:
            raise ValueError(
                f"Unknown preprocessing: {preprocessing}. Available: {list(PREPROCESSING_METHODS)}"
            )
        self.model_name = model_name
        self.preprocessing = preprocessing
        self.logger = logger
        self.model = self.MODELS[model_name]
        self.best_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.impute_values = None
        self.imputer = None

    @staticmethod
    def _coerce_features(X: pd.DataFrame) -> pd.DataFrame:
        """Convert feature table to numeric form, coercing comma-decimal strings."""
        out = X.copy()
        for col in out.columns:
            if out[col].dtype == object:
                cleaned = out[col].astype(str).str.replace(",", ".", regex=False)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
            else:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        return out
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except Exception as e:
                self.logger.warning(f"ROC-AUC failed: {e}")
                metrics["roc_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
        
        return metrics
    
    def train_with_cv(self, X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> Dict:
        """Train with cross-validation and grid search."""
        from sklearn.model_selection import GridSearchCV
        
        self.logger.info(f"Training {self.model_name} with {cv_folds}-fold CV...")

        X_scaled = self._prepare_features(X_train, fit=True)
        
        grid_search = GridSearchCV(
            self.model,
            self.PARAM_GRIDS[self.model_name],
            cv=cv_folds,
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_scaled, y_train)
        
        self.best_model = grid_search.best_estimator_
        
        self.logger.info(f"Best params: {grid_search.best_params_}")
        self.logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate on test set."""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_with_cv() first.")

        X_scaled = self._prepare_features(X_test, fit=False)
        y_pred = self.best_model.predict(X_scaled)
        
        if hasattr(self.best_model, "predict_proba"):
            y_pred_proba = self.best_model.predict_proba(X_scaled)
        else:
            y_pred_proba = None
        
        return self.compute_metrics(y_test.to_numpy(), y_pred, y_pred_proba)


    def _prepare_features(self, X: pd.DataFrame, fit: bool) -> np.ndarray:
        """Coerce to numeric, apply imputation/scaling, and optionally add missing indicators."""
        X_numeric = self._coerce_features(X)
        if fit:
            self.feature_columns = list(X_numeric.columns)
        elif self.feature_columns is not None:
            X_numeric = X_numeric.reindex(columns=self.feature_columns)

        add_indicator = self.preprocessing == "mice_indicator"
        indicator = None
        if add_indicator:
            indicator = X_numeric.isna().to_numpy().astype(float)

        if self.preprocessing == "median":
            if fit:
                self.impute_values = X_numeric.median(numeric_only=True)
                X_imputed = X_numeric.fillna(self.impute_values)
                X_scaled = self.scaler.fit_transform(X_imputed)
            else:
                X_imputed = X_numeric.fillna(self.impute_values)
                X_scaled = self.scaler.transform(X_imputed)
        else:
            try:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer

                if fit:
                    self.imputer = IterativeImputer(
                        random_state=RANDOM_STATE,
                        max_iter=10,
                        sample_posterior=False,
                    )
                    X_imputed = self.imputer.fit_transform(X_numeric)
                    X_scaled = self.scaler.fit_transform(X_imputed)
                else:
                    if self.imputer is None:
                        raise ValueError("MICE imputer not initialized")
                    X_imputed = self.imputer.transform(X_numeric)
                    X_scaled = self.scaler.transform(X_imputed)
            except Exception as exc:
                self.logger.warning(
                    f"MICE preprocessing failed ({exc}); falling back to median imputation."
                )
                if fit or self.impute_values is None:
                    self.impute_values = X_numeric.median(numeric_only=True)
                X_imputed = X_numeric.fillna(self.impute_values)
                X_scaled = self.scaler.fit_transform(X_imputed) if fit else self.scaler.transform(X_imputed)

        if indicator is not None:
            X_scaled = np.hstack([X_scaled, indicator])

        return X_scaled
