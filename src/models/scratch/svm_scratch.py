import numpy as np
from src.base import BaseClassifier
from src.dependencies import LinearKernel, PolynomialKernel, GaussianKernel


class SVMScratch(BaseClassifier): 

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "linear",
        gamma: float = 0.1,
        degree: int = 3,
        bias_kernel: float = 1.0,
        lr: float = 0.001,
        max_iter: int = 300,
        max_support_samples: int = 2000,
        random_state: int = 42,
    ):
        super().__init__()
        self.C = C
        self.kernel = kernel     # "linear", "poly", or "gaussian"
        self.gamma = gamma
        self.degree = degree
        self.bias_kernel = bias_kernel
        self.lr = lr
        self.max_iter = max_iter
        self.max_support_samples = max_support_samples
        self.random_state = random_state

        self.alphas_ = None
        self.support_vectors_ = None
        self.support_labels_ = None
        self.b_ = 0.0

    def _get_kernel(self):
        k = self.kernel.lower()
        if k == "linear":
            return LinearKernel()
        elif k in ("poly", "polynomial"):
            return PolynomialKernel(gamma=self.gamma, bias=self.bias_kernel, degree=self.degree)
        elif k in ("gaussian", "rbf"):
            return GaussianKernel(gamma=self.gamma)
        raise ValueError(f"Unknown kernel: {self.kernel}")

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self._get_kernel().compute(X1, X2)

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_raw = np.asarray(y)
        self.classes_ = np.unique(y_raw)
        y_signed = np.where(y_raw == self.classes_[1], 1.0, -1.0)

        # Subsample if dataset is huge
        if self.max_support_samples and len(X_arr) > self.max_support_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X_arr), size=self.max_support_samples, replace=False)
            X_arr, y_signed = X_arr[idx], y_signed[idx]

        n_samples = len(X_arr)
        K = self._kernel_matrix(X_arr, X_arr)
        alphas = np.zeros(n_samples, dtype=float)

        Y_K = np.outer(y_signed, y_signed) * K

        eff_lr = np.mininum(self.lr, 1.0 / (np.sum(np.abs(Y_K), axis=1) + 1e-8))
        for _ in range(self.max_iter):
            grad = 1.0 - np.dot(Y_K, alphas)
            alphas = np.clip(alphas + eff_lr * grad, 0.0, self.C)
            alphas -= y_signed * (np.dot(alphas, y_signed) / n_samples)
            alphas = np.clip(alphas, 0.0, self.C)

        sv_mask = alphas > 1e-5
        if not np.any(sv_mask):
            sv_mask[:] = True

        self.alphas_ = alphas[sv_mask]
        self.support_vectors_ = X_arr[sv_mask]
        self.support_labels_ = y_signed[sv_mask]

        # Compute bias b from support vectors on the margin
        K_sv = self._kernel_matrix(self.support_vectors_, self.support_vectors_)
        preds_sv = np.dot(K_sv, self.alphas_ * self.support_labels_)
        
        margin_mask = (self.alphas_ > 1e-5) & (self.alphas_ < self.C - 1e-5)
        if np.any(margin_mask):
            self.b_ = float(np.mean(self.support_labels_[margin_mask] - preds_sv[margin_mask]))
        else:
            self.b_ = float(np.mean(self.support_labels_ - preds_sv))

        self.is_fitted = True
        return self

    def decision_function(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        K_test = self._kernel_matrix(X_arr, self.support_vectors_)
        return np.dot(K_test, self.alphas_ * self.support_labels_) + self.b_

    def predict_prob(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])