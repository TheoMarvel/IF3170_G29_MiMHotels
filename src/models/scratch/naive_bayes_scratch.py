import numpy as np
from src.base import BaseClassifier


class NaiveBayesScratch(BaseClassifier): # Mixed Gaussian & Categorical Naive Bayes

    def __init__(self, laplace_alpha: float = 1.0, var_smoothing: float = 1e-8, max_cat_unique: int = 15):
        super().__init__()
        self.laplace_alpha = laplace_alpha
        self.var_smoothing = var_smoothing
        self.max_cat_unique = max_cat_unique

        self.log_priors_ = None
        self.is_categorical_ = []
        self.gaussian_params_ = {}  # {col_idx: {class_label: (mean, var)}}
        self.cat_cpts_ = {}         # {col_idx: {class_label: ({val: log_prob}, default_log_prob)}}

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        self.log_priors_ = np.log(counts / len(y_arr))

        n_features = X_arr.shape[1]
        self.is_categorical_ = []
        self.gaussian_params_ = {}
        self.cat_cpts_ = {}

        for j in range(n_features):
            col = X_arr[:, j]
            uniq_vals = np.unique(col)
            is_cat = len(uniq_vals) <= self.max_cat_unique
            self.is_categorical_.append(is_cat)

            if is_cat:
                self.cat_cpts_[j] = {}
                n_unique = len(uniq_vals)
                for c in self.classes_:
                    col_c = col[y_arr == c]
                    denom = len(col_c) + self.laplace_alpha * n_unique
                    val_log_probs = {}
                    for v in uniq_vals:
                        cnt = np.sum(col_c == v)
                        val_log_probs[v] = float(np.log((cnt + self.laplace_alpha) / denom))
                    # Unseen category fallback using Laplace smoothing (count = 0)
                    default_lp = float(np.log(self.laplace_alpha / denom))
                    self.cat_cpts_[j][c] = (val_log_probs, default_lp)
            else:
                self.gaussian_params_[j] = {}
                for c in self.classes_:
                    col_c = col[y_arr == c]
                    mean = float(np.mean(col_c))
                    var = float(np.var(col_c) + self.var_smoothing)
                    self.gaussian_params_[j][c] = (mean, var)

        self.is_fitted = True
        return self

    def _compute_log_Joint(self, X_arr: np.ndarray) -> np.ndarray:
        n_samples, n_features = X_arr.shape
        log_joint = np.tile(self.log_priors_, (n_samples, 1))

        for c_idx, c in enumerate(self.classes_):
            for j in range(n_features):
                col = X_arr[:, j]
                if self.is_categorical_[j]:
                    val_map, default_lp = self.cat_cpts_[j][c]
                    log_lik = np.array([val_map.get(v, default_lp) for v in col])
                else:
                    mean, var = self.gaussian_params_[j][c]
                    log_lik = -0.5 * np.log(2.0 * np.pi * var) - ((col - mean) ** 2) / (2.0 * var)
                log_joint[:, c_idx] += log_lik

        return log_joint

    def predict_prob(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=float)
        log_joint = self._compute_log_Joint(X_arr)
        # Log-sum-exp trick to convert log-joint to normalized probabilities safely
        max_log = np.max(log_joint, axis=1, keepdims=True)
        exp_scores = np.exp(log_joint - max_log)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        probs = self.predict_prob(X)
        return self.classes_[np.argmax(probs, axis=1)]