"""Baseline ML models: Random Forest and XGBoost."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# xgboost is imported lazily inside XGBoostModel. Its wheel links against a
# system OpenMP runtime (libomp on macOS) that is not installed everywhere, and
# a missing system library should not stop anyone from using the Random Forest
# baseline or the rest of this package.


class RandomForestModel:
    """Random Forest classifier for molecular property prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Train the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> np.ndarray:
        """Return feature importances."""
        return self.model.feature_importances_


class XGBoostModel:
    """XGBoost classifier for molecular property prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.scale_pos_weight = scale_pos_weight
        try:
            from xgboost import XGBClassifier
        except (ImportError, Exception) as exc:  # noqa: B014 - xgboost raises XGBoostError
            raise ImportError(
                "xgboost is unavailable. On macOS this usually means the OpenMP "
                "runtime is missing -- `brew install libomp`. The Random Forest "
                "and HistGradientBoosting baselines need no system libraries."
            ) from exc

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="logloss",
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        early_stopping_rounds: int | None = 10,
    ) -> "XGBoostModel":
        """
        Train the model with optional early stopping.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels (for early stopping)
            early_stopping_rounds: Stop if no improvement after this many rounds
        """
        # Auto-compute scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            self.model.set_params(scale_pos_weight=n_neg / n_pos)

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            if early_stopping_rounds:
                fit_params["early_stopping_rounds"] = early_stopping_rounds
                fit_params["verbose"] = False

        self.model.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> np.ndarray:
        """Return feature importances."""
        return self.model.feature_importances_
