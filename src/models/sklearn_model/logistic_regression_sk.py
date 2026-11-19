import numpy as np
from sklearn.linear_model import LogisticRegression
from src.base import BaseClassifier


class LogisticRegressionSklearn(BaseClassifier): # Scikit-learn wrapper for Logistic Regression

    def __init__(
        self,
        penalty: str = "elasticnet",
        lambda_val: float = 0.01,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        super().__init__()
        self.penalty = penalty.lower() if penalty else "none"
        self.lambda_val = lambda_val
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state

        C_val = 1.0 / max(self.lambda_val, 1e-12)
        kwargs = {
            "solver": "saga",
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

        if self.penalty == "none":
            kwargs["penalty"] = None
        elif self.penalty in ("l1", "l2"):
            kwargs["penalty"] = self.penalty
            kwargs["C"] = C_val
        elif self.penalty == "elasticnet":
            kwargs["penalty"] = "elasticnet"
            kwargs["C"] = C_val
            kwargs["l1_ratio"] = self.l1_ratio
        else:
            raise ValueError(f"Unknown penalty: {self.penalty}")

        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.model.fit(X_arr, y_arr)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def predict_prob(self, X) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict(np.asarray(X, dtype=float))