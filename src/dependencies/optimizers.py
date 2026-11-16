from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC): # Base class for gradient optimizers

    @abstractmethod
    def step(self, weights: np.ndarray, bias: float, grad_w: np.ndarray, grad_b: float):
        pass


class GradientOptimizer(Optimizer): # Gradient Descent / Ascent Optimizer

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, weights: np.ndarray, bias: float, grad_w: np.ndarray, grad_b: float):
        new_weights = weights - self.learning_rate * grad_w
        new_bias = bias - self.learning_rate * grad_b
        return new_weights, new_bias
