import numpy as np
import pandas as pd
from src.base import BaseTransformer


class FeatureScaler(BaseTransformer): # Scales numerical features using train statistics

    def __init__(self, method: str = "standard", columns: list = None):
        super().__init__()
        self.method = method  # "standard", "minmax", or "robust"
        self.columns = columns
        self.stats_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = pd.DataFrame(X)
        target_cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()
        self.stats_ = {}

        for col in target_cols:
            vals = df[col].to_numpy(dtype=float)
            if self.method == "standard":
                self.stats_[col] = (np.mean(vals), np.std(vals) + 1e-8)
            elif self.method == "minmax":
                c_min, c_max = np.min(vals), np.max(vals)
                self.stats_[col] = (c_min, (c_max - c_min) + 1e-8)
            elif self.method == "robust":
                q25, med, q75 = np.percentile(vals, [25, 50, 75])
                self.stats_[col] = (med, (q75 - q25) + 1e-8)
            else:
                raise ValueError(f"Unknown scaling method: {self.method}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        df = pd.DataFrame(X).copy()

        for col, (center, scale) in self.stats_.items():
            if col in df.columns:
                df[col] = (df[col].to_numpy(dtype=float) - center) / scale
        return df