from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC): # Base class for SVM kernels
    @abstractmethod
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        pass


class LinearKernel(Kernel): # K(x1, x2) = x1.T . x2, Source: GeeksforGeeks (2025)
    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.dot(X1, X2.T)


class PolynomialKernel(Kernel): # K(x1, x2) = (gamma * (x1.T . x2) + bias)**degree, Source: GeeksforGeeks (2025)
    def __init__(self, gamma: float = 1.0, bias: float = 1.0, degree: int = 3):
        self.gamma = gamma
        self.bias = bias
        self.degree = degree

    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return (self.gamma * np.dot(X1, X2.T) + self.bias) ** self.degree


class GaussianKernel(Kernel): # RBF Kernel: K(x1, x2) = exp(-gamma * ||x1 - x2||^2)
    def __init__(self, gamma: float = 0.1):
        self.gamma = gamma

    def compute(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        x1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        x2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
        squared_dist = np.maximum(x1_sq + x2_sq - 2 * np.dot(X1, X2.T), 0.0)
        return np.exp(-self.gamma * squared_dist)