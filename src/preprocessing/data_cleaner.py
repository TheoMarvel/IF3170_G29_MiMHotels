import numpy as np
import pandas as pd
from src.base import BaseTransformer


class DataCleaner(BaseTransformer): # Imputes missing values using train statistics

    def __init__(self, num_imputation: str = "median", cat_imputation: str = "mode"):
        super().__init__()
        self.num_imputation = num_imputation
        self.cat_imputation = cat_imputation
        self.fill_values_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        df = pd.DataFrame(X)
        self.fill_values_ = {}

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                if self.num_imputation == "mean":
                    self.fill_values_[col] = df[col].mean()
                else:
                    self.fill_values_[col] = df[col].median()
            else:
                mode_series = df[col].mode(dropna=True)
                self.fill_values_[col] = mode_series.iloc[0] if not mode_series.empty else "Unknown"

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        df = pd.DataFrame(X).copy()
        return df.fillna(self.fill_values_)

    @staticmethod
    def remove_duplicates(X: pd.DataFrame, y: pd.Series = None):
        # Only call on Training Data before fitting
        mask = ~X.duplicated()
        if y is not None:
            return X.loc[mask].reset_index(drop=True), y.loc[mask].reset_index(drop=True)
        return X.loc[mask].reset_index(drop=True)