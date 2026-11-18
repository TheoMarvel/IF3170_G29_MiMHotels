import numpy as np
from src.base import BaseClassifier
from src.dependencies import EuclideanDistance, ManhattanDistance, MinkowskiDistance


class KNN(BaseClassifier): # K-Nearest Neighbors classifier

    def __init__(self, k: int = 3, distance: str = "euclidean", p: float = 2.0, weights: str = "uniform", batch_size: int = 200):
        super().__init__()
        self.k = k
        self.distance = distance   # "euclidean", "manhattan", or "minkowski"
        self.p = p
        self.weights = weights    
        self.batch_size = batch_size
        self.X_train_ = None
        self.y_train_ = None

    def _get_distance(self):
        if self.distance == "euclidean":
            return EuclideanDistance()
        elif self.distance == "manhattan":
            return ManhattanDistance()
        elif self.distance == "minkowski":
            return MinkowskiDistance(p=self.p)
        raise ValueError(f"Unknown distance metric: {self.distance}")

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.unique(self.y_train_)
        self.is_fitted = True
        return self

    def predict_prob(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_test = np.asarray(X, dtype=float)
        distance_metric = self._get_distance()
        n_samples = X_test.shape[0]
        probs = np.zeros((n_samples, len(self.classes_)), dtype=float)

        # Process in batches
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            dist_batch = distance_metric.calculate(X_test[start:end], self.X_train_)

            # Find indices of k nearest neighbors
            knn_idx = np.argpartition(dist_batch, kth=self.k - 1, axis=1)[:, :self.k]
            knn_labels = self.y_train_[knn_idx]

            if self.weights == "distance":
                knn_dists = np.take_along_axis(dist_batch, knn_idx, axis=1)
                w = 1.0 / (knn_dists + 1e-8)
            else:
                w = np.ones_like(knn_labels, dtype=float)

            total_w = np.sum(w, axis=1, keepdims=True)
            for idx, c in enumerate(self.classes_):
                probs[start:end, idx] = np.sum(w * (knn_labels == c), axis=1) / total_w[:, 0]

        return probs

    def predict(self, X) -> np.ndarray:
        probs = self.predict_prob(X)
        return self.classes_[np.argmax(probs, axis=1)]