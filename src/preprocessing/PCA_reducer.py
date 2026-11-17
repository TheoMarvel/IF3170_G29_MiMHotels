import numpy as np
from src.base import BaseTransformer


class PCAReducer(BaseTransformer): # Dimensionality reduction using PCA

    def __init__(self, n_components: float = 2):
        super().__init__()
        self.n_components = n_components  # int (count) or float in (0, 1) for variance ratio
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X_arr, axis=0)
        X_centered = X_arr - self.mean_

        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]

        total_var = np.sum(eigenvalues) + 1e-12
        var_ratios = eigenvalues / total_var

        if 0 < self.n_components < 1.0:
            cum_var = np.cumsum(var_ratios)
            k = int(np.searchsorted(cum_var, self.n_components) + 1)
        else:
            k = int(self.n_components)

        self.components_ = eigenvectors[:, :k]
        self.explained_variance_ratio_ = var_ratios[:k]
        self.is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        return np.dot(X_arr - self.mean_, self.components_)