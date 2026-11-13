from abc import ABC, abstractmethod
import numpy as np
from src.base.serializer import Serializer


class BaseEstimator(ABC): # Base class for all transformers and classifiers
    
    def __init__(self):
        self.is_fitted = False

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} not fitted to training data.")

    def save(self, filepath: str):
        Serializer.save(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        return Serializer.load(filepath)


class BaseTransformer(BaseEstimator, ABC): # Class for preprocessing data by transformation

    @abstractmethod
    def fit(self, X, y=None):
        pass

    @abstractmethod
    def transform(self, X):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class BaseClassifier(BaseEstimator, ABC): # Class for supervised learning models
    
    def __init__(self):
        super().__init__()
        self.classes_ = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_prob(self, X):
        pass

    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return float(np.mean(np.asarray(y) == y_pred))