import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from src.base import BaseClassifier


class NaiveBayesSklearn(BaseClassifier): # Scikit-learn wrapper for Naive Bayes

    def __init__(self, nb_type: str = "gaussian", var_smoothing: float = 1e-8, alpha: float = 1.0):
        super().__init__()
        self.nb_type = nb_type.lower()  # "gaussian" or "bernoulli"
        self.var_smoothing = var_smoothing
        self.alpha = alpha

        if self.nb_type == "gaussian":
            self.model = GaussianNB(var_smoothing=self.var_smoothing)
        elif self.nb_type == "bernoulli":
            self.model = BernoulliNB(alpha=self.alpha)
        else:
            raise ValueError(f"Unknown Naive Bayes type: {self.nb_type}")

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