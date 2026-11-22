import numpy as np
import pandas as pd
from src.base import BaseClassifier
from src.dependencies import InformationGain, GainRatio, DecisionTreeNode


class DecisionTreeScratch(BaseClassifier): # DTL ID3 and C4.5 supporting numeric, categorical, and null values

    def __init__(self, algorithm: str = "C4.5", max_depth: int = 10, min_samples_split: int = 5, min_gain: float = 1e-4, post_prune: bool = False, random_state: int=None):
        super().__init__()
        self.algorithm = algorithm  # "ID3" or "C4.5"
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.post_prune = post_prune
        self.random_state = random_state
        self.root_ = None
        self.is_continuous_ = []

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=object)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.criterion_ = GainRatio() if self.algorithm.upper() == "C4.5" else InformationGain()

        # Column is continuous if it is numeric and has > 10 unique values
        self.is_continuous_ = []
        for j in range(X_arr.shape[1]):
            valid = X_arr[~pd.isna(X_arr[:, j]), j]
            is_num = len(valid) > 0 and isinstance(valid[0], (int, float, np.number))
            self.is_continuous_.append(is_num and len(np.unique(valid)) > 10)

        if self.post_prune and len(y_arr) >= 20:
            rng = np.random.default_rng(self.random_state)
            idx = rng.permutation(len(y_arr))
            n_prune = int(0.15 * len(y_arr))
            prune_idx, grow_idx = idx[:n_prune], idx[n_prune:]
            self.root_ = self._build_tree(X_arr[prune_idx], y_arr[grow_idx], depth=0)
            self._prune_node(self.root_, X_arr[prune_idx], y_arr[prune_idx])
        else:
            self.root_ = self._build_tree(X_arr, y_arr, depth=0)

        self.is_fitted = True
        return self

    def _make_branches(self, col_valid: np.ndarray, is_cont: bool, thresh=None) -> dict:
        if is_cont:
            vals = col_valid.astype(float)
            return {"left": vals <= thresh, "right": vals > thresh}
        return {cat: (col_valid == cat) for cat in np.unique(col_valid)}

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain, best_feat, best_thresh = -1.0, None, None

        for j in range(X.shape[1]):
            col = X[:, j]
            valid_mask = ~pd.isna(col)
            if np.sum(valid_mask) < self.min_samples_split:
                continue

            col_valid, y_valid = col[valid_mask], y[valid_mask]
            is_cont = self.is_continuous_[j]

            if is_cont:
                # Test 10 quantile thresholds for continuous features
                candidates = np.unique(np.percentile(col_valid.astype(float), np.linspace(10, 90, 9)))
            else:
                candidates = [None] if len(np.unique(col_valid)) > 1 else []

            for thresh in candidates:
                branches = self._make_branches(col_valid, is_cont, thresh)
                y_splits = [y_valid[mask] for mask in branches.values() if np.any(mask)]
                if len(y_splits) <= 1:
                    continue

                gain = self.criterion_.compute_score(y_valid, y_splits)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, j, thresh

        return best_feat, best_thresh, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        probs = np.array([np.mean(y == c) for c in self.classes_])
        pred_class = self.classes_[np.argmax(probs)]

        if len(np.unique(y)) <= 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return DecisionTreeNode(class_probs=probs, predicted_class=pred_class, is_leaf=True)

        feat_idx, thresh, gain = self._best_split(X, y)
        if feat_idx is None or gain < self.min_gain:
            return DecisionTreeNode(class_probs=probs, predicted_class=pred_class, is_leaf=True)

        is_cont = self.is_continuous_[feat_idx]
        node = DecisionTreeNode(
            feature_idx=feat_idx, threshold=thresh, is_continuous=is_cont,
            gain_score=gain, class_probs=probs, predicted_class=pred_class, is_leaf=False
        )

        col = X[:, feat_idx]
        valid_mask = ~pd.isna(col)
        branches = self._make_branches(col[valid_mask], is_cont, thresh)

        # Find majority branch to route null rows during training 
        majority_key = max(branches, key=lambda k: np.sum(branches[k]))

        for key, valid_branch_mask in branches.items():
            full_mask = np.zeros(len(y), dtype=bool)
            full_mask[valid_mask] = valid_branch_mask
            if key == majority_key:
                full_mask[~valid_mask] = True  # Assign null rows to majority branch

            if np.any(full_mask):
                node.children[key] = self._build_tree(X[full_mask], y[full_mask], depth + 1)

        return node

    def _prune_node(self, node: DecisionTreeNode, X_sub: np.ndarray, y_sub:np.ndarray):
        if node.is_leaf or not node.children or len(y_sub) == 0:
            return

        # 1. Bottom-up: prune all child subtrees first
        col = X_sub[:, node.feature_idx]
        valid = ~pd.isna(col)
        for key, child in node.children.items():
            if node.is_continuous:
                mask = valid & ((col.astype(float) <= node.threshold) if key =="left" else (col.astype(float) > node.threshold))
            else:
                mask = valid & (col == key)

            if np.any(mask):
                self._prune_node(child, X_sub[mask], y_sub[mask])

        # 2. Compare subtree accuracy vs. leaf accuracy on rows reaching this node
        subtree_preds = np.array([self.classes_[np.argmax(self._predict_row(r, node))] for r in X_sub])
        acc_subtree = np.mean(subtree_preds == y_sub)
        acc_leaf = np.mean(node.predicted_class == y_sub)

        if acc_leaf >= acc_subtree:
            node.is_leaf = True
            node.children = {}

    def prune(self, X_prune, y_prune):
        self._check_is_fitted()
        self._prune_node(self.root_, np.asarray(X_prune, dtype=object), np.asarray(y_prune))
        return self    
    
    def _predict_row(self, row: np.ndarray, node: DecisionTreeNode) -> np.ndarray:
        if node.is_leaf or not node.children:
            return node.class_probs

        val = row[node.feature_idx]
        if pd.isna(val):
            return node.class_probs

        key = ("left" if float(val) <= node.threshold else "right") if node.is_continuous else val
        if key in node.children:
            return self._predict_row(row, node.children[key])
        return node.class_probs

    def predict_prob(self, X) -> np.ndarray:
        self._check_is_fitted()
        X_arr = np.asarray(X, dtype=object)
        return np.array([self._predict_row(row, self.root_) for row in X_arr])

    def predict(self, X) -> np.ndarray:
        probs = self.predict_prob(X)
        return self.classes_[np.argmax(probs, axis=1)]