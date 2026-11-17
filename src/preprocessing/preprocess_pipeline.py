from src.base import BaseTransformer


class PreprocessingPipeline(BaseTransformer): # Chains multiple preprocessing transformers

    def __init__(self, steps: list):
        super().__init__()
        self.steps = steps  # List of BaseTransformer instances

    def fit(self, X, y=None):
        data = X
        for step in self.steps:
            data = step.fit(data, y)
        self.is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        data = X
        for step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(self, X, y=None):
        data = X
        for step in self.steps:
            data = step.fit_transform(data, y)
        self.is_fitted = True
        return data