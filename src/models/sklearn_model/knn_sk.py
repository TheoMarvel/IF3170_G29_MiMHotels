import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from src.base import BaseClassifier


class KNNSklearn(BaseClassifier): # KNN using sklearn

    def __init__(self, k: int = 3, distance: str = "euclidean", p: float = 2.0, weights: str = "uniform"):
        super().__init__()
        self.k = k
        self.distance = distance  # "euclidean", "manhattan", or "minkowski"
        self.p = p
        self.weights = weights
        self.model = KNeighborsClassifier(
            n_neighbors=self.k,
            metric=self.distance,
            p=self.p,
            weights=self.weights,
        )

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