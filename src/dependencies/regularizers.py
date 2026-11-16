from abc import ABC, abstractmethod
import numpy as np


class Regularizer(ABC): # Base class for weight regularization
    @abstractmethod
    def penalty(self, weights: np.ndarray) -> float:
        pass

    @abstractmethod
    def gradient(self, weights: np.ndarray) -> np.ndarray:
        pass


class NoRegularizer(Regularizer): # unregularized training
    def penalty(self, weights: np.ndarray) -> float:
        return 0.0

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return np.zeros_like(weights)


class L1Regularizer(Regularizer): # Lasso: lambda * sum(|w|)
    def __init__(self, lambda_val: float = 0.01):
        self.lambda_val = lambda_val

    def penalty(self, weights: np.ndarray) -> float:
        return float(self.lambda_val * np.sum(np.abs(weights)))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.lambda_val * np.sign(weights)


class L2Regularizer(Regularizer): # Ridge: (lambda / 2) * sum(w^2)
    def __init__(self, lambda_val: float = 0.01):
        self.lambda_val = lambda_val

    def penalty(self, weights: np.ndarray) -> float:
        return float(0.5 * self.lambda_val * np.sum(weights ** 2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return self.lambda_val * weights


class ElasticNetRegularizer(Regularizer): # Combination of L1 and L2 Regularization: lambda * ((l1_ratio*sum(|w|) + (((1-l1_ratio)/2)*sum(w^2)))

    def __init__(self, lambda_val: float = 0.01, l1_ratio: float = 0.5):
        self.lambda_val = lambda_val
        self.l1_ratio = l1_ratio

    def penalty(self, weights: np.ndarray) -> float:
        l1 = np.sum(np.abs(weights))
        l2 = 0.5 * np.sum(weights ** 2)
        return float(self.lambda_val * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        l1_grad = np.sign(weights)
        l2_grad = weights
        return self.lambda_val * (self.l1_ratio * l1_grad + (1.0 - self.l1_ratio) * l2_grad)