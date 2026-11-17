import numpy as np
import pandas as pd
from src.base import BaseTransformer


class FeatureSelector(BaseTransformer): # Selects features based on variance and target correlation

    def __init__(self, min_variance: float = 1e-5, min_corr: float = 0.01, top_k: int = None):
        super().__init__()
        self.min_variance = min_variance
        self.min_corr = min_corr
        self.top_k = top_k
        self.selected_features_ = []
        self.feature_scores_ = {}

    def fit(self, X: pd.DataFrame, y):
        df = pd.DataFrame(X)
        y_array = np.asarray(y, dtype=float)
        self.feature_scores_ = {}

        for col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            if np.var(vals) <= self.min_variance:
                continue
            # Absolute Pearson correlation |r| with target y
            corr = np.abs(np.corrcoef(vals, y_array)[0, 1])
            if not np.isnan(corr) and corr >= self.min_corr:
                self.feature_scores_[col] = float(corr)

        sorted_cols = sorted(self.feature_scores_, key=self.feature_scores_.get, reverse=True)
        self.selected_features_ = sorted_cols[:self.top_k] if self.top_k else sorted_cols
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        df = pd.DataFrame(X)
        return df[self.selected_features_].copy()