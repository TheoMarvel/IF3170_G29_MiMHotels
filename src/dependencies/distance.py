from abc import ABC, abstractmethod
import numpy as np


class DistanceMetric(ABC): # Base class for distance calculations in KNN
    @abstractmethod
    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        pass

class ManhattanDistance(DistanceMetric): # L1 distance: sum(|x1 - x2|)
    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(X1[:, None, :] - X2[None, :, :]), axis=2)

class EuclideanDistance(DistanceMetric): # L2 distance: sqrt(sum((x1 - x2)^2))
    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Using (a - b)^2 = a^2 + b^2 - 2ab for fast matrix calculation
        x1_square = np.sum(X1 ** 2, axis=1, keepdims=True)
        x2_square = np.sum(X2 ** 2, axis=1, keepdims=True).T
        square_dist = np.maximum(x1_square + x2_square - 2 * np.dot(X1, X2.T), 0.0)
        return np.sqrt(square_dist)

class MinkowskiDistance(DistanceMetric): 
    def __init__(self, p: float = 2.0):
        if p <= 0:
            raise ValueError("Parameter p must be greater than 0.")
        self.p = p

    def calculate(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = np.abs(X1[:, None, :] - X2[None, :, :])
        return np.sum(diff ** self.p, axis=2) ** (1.0 / self.p)
