import numpy as np
from sklearn.svm import SVC
from src.base import BaseClassifier


class SVMSklearn(BaseClassifier): # Scikit-learn wrapper for Kernel SVM

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        gamma: float = 0.1,
        degree: int = 3,
        bias_kernel: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        super().__init__()
        self.C = C
        self.kernel = kernel.lower()
        self.gamma = gamma
        self.degree = degree
        self.bias_kernel = bias_kernel
        self.max_iter = max_iter
        self.random_state = random_state

        kernel_map = {
            "linear": "linear",
            "poly": "poly",
            "polynomial": "poly",
            "gaussian": "rbf",
            "rbf": "rbf",
        }
        if self.kernel not in kernel_map:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        self.model = SVC(
            C=self.C,
            kernel=kernel_map[self.kernel],
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.bias_kernel,
            probability=True,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.model.fit(X_arr, y_arr)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self

    def decision_function(self, X) -> np.ndarray:
        self._check_is_fitted()
        return self.model.decision_function(np.asarray(X, dtype=float))

    def predict_prob(self, X) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict_proba(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        self._check_is_fitted()
        return self.model.predict(np.asarray(X, dtype=float))