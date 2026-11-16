from typing import Any, Dict, Optional
import numpy as np


class DecisionTreeNode: # Represents an internal split node or leaf node in DTL

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[ Any ] = None,
        is_continuous: bool = False,
        gain_score: float = 0.0,
        class_probs: Optional[np.ndarray] = None,
        predicted_class: Optional[Any] = None,
        is_leaf: bool = False,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold          # Split value (c) for continuous or binary categorical
        self.is_continuous = is_continuous
        self.gain_score = gain_score        # Used to filter top-N branches
        self.class_probs = class_probs      # Probability array for each class at this node
        self.predicted_class = predicted_class
        self.is_leaf = is_leaf

        self.children: Dict[Any, "DecisionTreeNode"] = {}
        self.branch_weights: Dict[Any, float] = {} # Proportions for routing null values in  C4.5