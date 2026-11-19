import numpy as np
from src.base import BaseClassifier
from src.dependencies import (
    NoRegularizer,
    L1Regularizer,
    L2Regularizer,
    ElasticNetRegularizer,
    GradientOptimizer
)


class LogisticRegressionScratch(BaseClassifier): # Logistic Regression with L1, L2, and Elastic Net

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_iter: int = 1000,
        penalty: str = "elasticnet",
        lambda_val: float = 0.01,
        l1_ratio: float = 0.5,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.penalty = penalty          # "none", "l1", "l2", or "elasticnet"
        self.lambda_val = lambda_val
        self.l1_ratio = l1_ratio
        self.batch_size = batch_size
        self.random_state = random_state

        self.weights_ = None
        self.bias_ = 0.0
        self.loss_history_ = []
        self.trajectory_ = []           # Stores (theta_0, theta_1)

    def _get_regularizer(self):
        p = self.penalty.lower() if self.penalty else "none"
        if p == "none":
            return NoRegularizer()
        elif p == "l1":
            return L1Regularizer(self.lambda_val)
        elif p == "l2":
            return L2Regularizer(self.lambda_val)
        elif p == "elasticnet":
            return ElasticNetRegularizer(self.lambda_val, self.l1_ratio)
        raise ValueError(f"Unknown penalty: {self.penalty}")

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _compute_loss(self, X: np.ndarray, y: np.ndarray, reg) -> float:
        probs = self._sigmoid(np.dot(X, self.weights_) + self.bias_)
        eps = 1e-12
        bce = -np.mean(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps))
        return float(bce + reg.penalty(self.weights_))

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_raw = np.asarray(y)
        self.classes_ = np.unique(y_raw)
        y_bin = (y_raw == self.classes_[1]).astype(float)

        n_samples, n_features = X_arr.shape
        self.weights_ = np.zeros(n_features, dtype=float)
        self.bias_ = 0.0
        self.loss_history_ = []
        self.trajectory_ = []

        reg = self._get_regularizer()
        optimizer = GradientOptimizer(learning_rate=self.learning_rate)
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.max_iter):
            if self.batch_size and self.batch_size < n_samples:
                idx = rng.choice(n_samples, size=self.batch_size, replace=False)
                X_b, y_b = X_arr[idx], y_bin[idx]
            else:
                X_b, y_b = X_arr, y_bin

            preds = self._sigmoid(np.dot(X_b, self.weights_) + self.bias_)
            error = preds - y_b

            grad_w = (np.dot(X_b.T, error) / len(y_b)) + reg.gradient(self.weights_)
            grad_b = float(np.mean(error))

            self.weights_, self.bias_ = optimizer.step(self.weights_, self.bias_, grad_w, grad_b)
            self.loss_history_.append(self._compute_loss(X_arr, y_bin, reg))
            if n_features >= 1:
                self.trajectory_.append((float(self.bias_), float(self.weights_[0])))

        self.is_fitted = True
        return self

    def predict_prob(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        p1 = self._sigmoid(np.dot(X_arr, self.weights_) + self.bias_)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        probs = self.predict_prob(X)
        return self.classes_[np.argmax(probs, axis=1)]