import numpy as np
import pandas as pd
from src.base import BaseTransformer


class CategoricalEncoder(BaseTransformer): # Give label or One-Hot encoding for categorical features

    def __init__(self, method: str = "label", columns: list = None):
        super().__init__()
        self.method = method  # "label" or "onehot"
        self.columns = columns
        self.mappings_ = {}
        self.onehot_cols_ = []

    def fit(self, X: pd.DataFrame, y=None):
        df = pd.DataFrame(X)
        target_cols = self.columns or df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.mappings_ = {}

        for col in target_cols:
            unique_vals = df[col].dropna().unique()
            self.mappings_[col] = {val: idx for idx, val in enumerate(unique_vals)}

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        df = pd.DataFrame(X).copy()

        if self.method == "label":
            for col, mapping in self.mappings_.items():
                if col in df.columns:
                    df[col] = df[col].map(mapping).fillna(-1).astype(int) # Unseen categories in test data are mapped to -1
            return df

        elif self.method == "onehot":
            for col, mapping in self.mappings_.items():
                if col not in df.columns:
                    continue
                for val in mapping.keys():
                    df[f"{col}_{val}"] = (df[col] == val).astype(int)
                df.drop(columns=[col], inplace=True)
            return df

        raise ValueError(f"Unknown encoding method: {self.method}")