"""
Predictive models and probability calibration module.
Implements Prevalence, Activity Dormancy Rule, Logistic Regression, LightGBM, XGBoost,
along with Platt Scaling and Isotonic Regression calibrators.
"""
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb

class PrevalenceBaseline:
    """Constant prior baseline matching training class prevalence."""
    def __init__(self):
        self.prevalence = 0.16

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.prevalence = float(np.mean(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        prob = np.full(shape=(len(X),), fill_value=self.prevalence)
        return prob

class ActivityDormancyRule:
    """
    Heuristic baseline that flags students purely based on days of inactivity
    (days_since_last_click), normalized to [0, 1].
    """
    def __init__(self, feature_idx: int = 18):
        self.feature_idx = feature_idx
        self.min_val = 0.0
        self.max_val = 60.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        dormancy = X[:, self.feature_idx]
        self.min_val = float(np.percentile(dormancy, 1))
        self.max_val = float(np.percentile(dormancy, 99))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dormancy = X[:, self.feature_idx]
        norm = (dormancy - self.min_val) / (self.max_val - self.min_val + 1e-5)
        return np.clip(norm, 0.0, 1.0)

class LogisticRegressionPipeline:
    """Standardized Logistic Regression with L2 regularization."""
    def __init__(self, C: float = 1.0, random_state: int = 42):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=1000, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)[:, 1]

class LightGBMPipeline:
    """Gradient Boosted Decision Trees via LightGBM."""
    def __init__(self, random_state: int = 42, **kwargs):
        params = {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1
        }
        params.update(kwargs)
        self.model = lgb.LGBMClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class XGBoostPipeline:
    """Extreme Gradient Boosting via XGBoost."""
    def __init__(self, random_state: int = 42, **kwargs):
        params = {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "eval_metric": "logloss",
            "n_jobs": -1
        }
        params.update(kwargs)
        self.model = xgb.XGBClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class PlattCalibrator:
    """Post-hoc probability calibration using Platt scaling (sigmoid logit fit)."""
    def __init__(self):
        self.lr = LogisticRegression(C=1.0)
        self.eps = 1e-7

    def fit(self, val_prob: np.ndarray, y_val: np.ndarray):
        val_p = np.clip(val_prob, self.eps, 1.0 - self.eps)
        val_logit = np.log(val_p / (1.0 - val_p)).reshape(-1, 1)
        self.lr.fit(val_logit, y_val)
        return self

    def transform(self, test_prob: np.ndarray) -> np.ndarray:
        test_p = np.clip(test_prob, self.eps, 1.0 - self.eps)
        test_logit = np.log(test_p / (1.0 - test_p)).reshape(-1, 1)
        return self.lr.predict_proba(test_logit)[:, 1]

class IsotonicCalibrator:
    """Post-hoc non-parametric probability calibration using Isotonic Regression."""
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, val_prob: np.ndarray, y_val: np.ndarray):
        self.iso.fit(val_prob, y_val)
        return self

    def transform(self, test_prob: np.ndarray) -> np.ndarray:
        cal = self.iso.predict(test_prob)
        return np.clip(cal, 0.0, 1.0)

