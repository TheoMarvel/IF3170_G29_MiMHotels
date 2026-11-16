from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class SplitCriteria(ABC): # Base class for Decision Tree splitting criteria

    @abstractmethod
    def entropy(self, y: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        pass

    @abstractmethod
    def compute_score(
        self,
        y_parent: np.ndarray,
        y_splits: List[np.ndarray],
        w_parent: Optional[np.ndarray] = None,
        w_splits: Optional[List[np.ndarray]] = None,
    ) -> float:
        pass

    def _class_probs(self, y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        if len(y) == 0:
            return np.array([])
        classes = np.unique(y)
        if weights is None:
            counts = np.array([np.sum(y == c) for c in classes], dtype=float)
            return counts / len(y)
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return np.array([])
        weighted_counts = np.array([np.sum(weights[y == c]) for c in classes], dtype=float)
        return weighted_counts / total_weight


class InformationGain(SplitCriteria): # Used by ID3 (Information Gain)

    def entropy(self, y: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        probs = self._class_probs(y, weights)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def compute_score(
        self,
        y_parent: np.ndarray,
        y_splits: List[np.ndarray],
        w_parent: Optional[np.ndarray] = None,
        w_splits: Optional[List[np.ndarray]] = None,
    ) -> float:
        parent_entropy = self.entropy(y_parent, w_parent)
        total_weight = float(len(y_parent)) if w_parent is None else float(np.sum(w_parent))
        if total_weight <= 0:
            return 0.0

        child_entropy = 0.0
        for i, y_child in enumerate(y_splits):
            if len(y_child) == 0:
                continue
            w_child = None if w_splits is None else w_splits[i]
            child_weight = float(len(y_child)) if w_child is None else float(np.sum(w_child))
            child_entropy += (child_weight / total_weight) * self.entropy(y_child, w_child)

        return parent_entropy - child_entropy


class GainRatio(InformationGain): # Used by C4.5 (Information Gain / Split Information)

    def compute_score(
        self,
        y_parent: np.ndarray,
        y_splits: List[np.ndarray],
        w_parent: Optional[np.ndarray] = None,
        w_splits: Optional[List[np.ndarray]] = None,
    ) -> float:
        infogain = super().compute_score(y_parent, y_splits, w_parent, w_splits)
        if infogain <= 0:
            return 0.0

        total_weight = float(len(y_parent)) if w_parent is None else float(np.sum(w_parent))
        split_info = 0.0
        for i, y_child in enumerate(y_splits):
            if len(y_child) == 0:
                continue
            w_child = None if w_splits is None else w_splits[i]
            child_weight = float(len(y_child)) if w_child is None else float(np.sum(w_child))
            ratio = child_weight / total_weight
            if ratio > 0:
                split_info -= ratio * np.log2(ratio)

        if split_info <= 1e-12:
            return 0.0
        return float(infogain / split_info)
