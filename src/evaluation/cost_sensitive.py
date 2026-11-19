import numpy as np
from src.base import BaseClassifier
from src.evaluation.metrics import Metrics


class CostSensitiveWrapper(BaseClassifier): # For Bayes Model
    def __init__(self, base_model: BaseClassifier, fp_cost: float = 2.0, fn_cost: float = 1.0, threshold: float = None):
        super().__init__()
        self.base_model = base_model
        self.fp_cost = fp_cost
        self.fn_cost = fn_cost
        # Theoretical Bayes optimal threshold: C(FP) / (C(FP) + C(FN)) = 2 / 3 ~= 0.667
        self.threshold = threshold if threshold is not None else fp_cost / (fp_cost + fn_cost)

    def fit(self, X, y):
        self.base_model.fit(X, y)
        self.classes_ = self.base_model.classes_
        self.is_fitted = True
        return self

    def tune_threshold(self, X_val, y_val, grid_size: int = 81) -> float:
        # Optional: empirically search for the threshold that minimizes validation cost
        probs = self.predict_prob(X_val)[:, 1]
        candidates = np.linspace(0.1, 0.9, grid_size)
        best_thresh = self.threshold
        best_cost = float("inf")

        for t in candidates:
            preds = np.where(probs >= t, self.classes_[1], self.classes_[0])
            cost = Metrics.total_cost(y_val, preds, self.fp_cost, self.fn_cost)
            if cost < best_cost:
                best_cost = cost
                best_thresh = float(t)

        self.threshold = best_thresh
        return self.threshold

    def predict_prob(self, X) -> np.ndarray:
        self._check_is_fitted()
        return self.base_model.predict_prob(X)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_prob(X)[:, 1]
        return np.where(probs >= self.threshold, self.classes_[1], self.classes_[0])