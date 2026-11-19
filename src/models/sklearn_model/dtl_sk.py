import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.base import BaseClassifier


class DecisionTreeSklearn(BaseClassifier): # Scikit-learn wrapper for ID3 and C4.5

    def __init__(
        self,
        algorithm: str = "C4.5",
        max_depth: int = 10,
        min_samples_split: int = 5,
        ccp_alpha: float = 0.005,
        random_state: int = 42,
    ):
        super().__init__()
        self.algorithm = algorithm.upper()  # "ID3" or "C4.5"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

        # Follow Cakupan #6 (Page 11) parameter requirements
        if self.algorithm == "ID3":
            criterion, alpha = "entropy", 0.0
        elif self.algorithm == "C4.5":
            criterion, alpha = "entropy", self.ccp_alpha
        else:
            raise ValueError(f"Unknown DTL algorithm: {self.algorithm}")

        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            ccp_alpha=alpha,
            random_state=self.random_state,
        )

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