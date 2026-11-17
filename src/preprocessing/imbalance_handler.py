import numpy as np
from src.base import BaseEstimator
from src.dependencies import EuclideanDistance


class ImbalanceHandler(BaseEstimator): # Resamples training data to balance class distribution

    def __init__(self, method: str = "smote", k_neighbors: int = 5, random_state: int = 42):
        super().__init__()
        self.method = method  # "undersample", "oversample", or "smote"
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        classes, counts = np.unique(y_arr, return_counts=True)
        if len(classes) != 2:
            return X_arr, y_arr

        min_class = classes[np.argmin(counts)]
        maj_class = classes[np.argmax(counts)]
        min_idx = np.where(y_arr == min_class)[0]
        maj_idx = np.where(y_arr == maj_class)[0]

        if self.method == "undersample":
            chosen_maj = rng.choice(maj_idx, size=len(min_idx), replace=False)
            idx = np.concatenate([min_idx, chosen_maj])
            rng.shuffle(idx)
            self.is_fitted = True
            return X_arr[idx], y_arr[idx]

        elif self.method == "oversample":
            extra_min = rng.choice(min_idx, size=len(maj_idx) - len(min_idx), replace=True)
            idx = np.concatenate([maj_idx, min_idx, extra_min])
            rng.shuffle(idx)
            self.is_fitted = True
            return X_arr[idx], y_arr[idx]

        elif self.method == "smote":
            n_synth = len(maj_idx) - len(min_idx)
            if n_synth <= 0:
                return X_arr, y_arr

            X_min = X_arr[min_idx]
            dist_matrix = EuclideanDistance().calculate(X_min, X_min)
            np.fill_diagonal(dist_matrix, np.inf)

            k = min(self.k_neighbors, len(X_min) - 1)
            nn_indices = np.argsort(dist_matrix, axis=1)[:, :k]

            synthetic = []
            for _ in range(n_synth):
                i = rng.integers(0, len(X_min))
                nn = nn_indices[i, rng.integers(0, k)] # nn = nearest neighbor
                gap = rng.random()
                synthetic.append(X_min[i] + gap * (X_min[nn] - X_min[i]))

            X_res = np.vstack([X_arr, np.array(synthetic)])
            y_res = np.concatenate([y_arr, np.full(n_synth, min_class)])
            self.is_fitted = True
            return X_res, y_res

        raise ValueError(f"Unknown imbalance method: {self.method}")