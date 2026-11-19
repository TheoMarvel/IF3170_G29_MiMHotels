import numpy as np
import pandas as pd
from src.evaluation.metrics import Metrics


class Validator: # Stratified train-val split and K-Fold cross-validation

    @staticmethod
    def train_val_split(X, y, val_size: float = 0.2, random_state: int = 42):
        rng = np.random.default_rng(random_state)
        y_arr = np.asarray(y).ravel()
        train_idx, val_idx = [], []

        # Split each class proportionally to preserve class balance
        for c in np.unique(y_arr):
            c_idx = np.where(y_arr == c)[0]
            rng.shuffle(c_idx)
            n_val = int(len(c_idx) * val_size)
            val_idx.extend(c_idx[:n_val])
            train_idx.extend(c_idx[n_val:])

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
        else:
            X_train, X_val = X[train_idx], X[val_idx]

        if isinstance(y, pd.Series):
            y_train = y.iloc[train_idx].reset_index(drop=True)
            y_val = y.iloc[val_idx].reset_index(drop=True)
        else:
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        return X_train, X_val, y_train, y_val

    @staticmethod
    def cross_validate(
        model_factory,
        X,
        y,
        k_folds: int = 5,
        pipeline_factory=None,
        balancer=None,
        random_state: int = 42,
    ) -> dict:
        rng = np.random.default_rng(random_state)
        y_arr = np.asarray(y).ravel()
        folds = [[] for _ in range(k_folds)]

        # Distribute each class evenly across k folds (Stratified K-Fold)
        for c in np.unique(y_arr):
            c_idx = np.where(y_arr == c)[0]
            rng.shuffle(c_idx)
            for i, idx in enumerate(c_idx):
                folds[i % k_folds].append(idx)

        fold_metrics = []
        for k in range(k_folds):
            val_idx = np.array(folds[k])
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != k])

            X_tr = X.iloc[train_idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[train_idx]
            X_va = X.iloc[val_idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else X[val_idx]
            y_tr, y_va = y_arr[train_idx], y_arr[val_idx]

            # Fit preprocessing strictly inside the training fold to prevent data leakage
            if pipeline_factory is not None:
                pipe = pipeline_factory()
                X_tr = pipe.fit_transform(X_tr, y_tr)
                X_va = pipe.transform(X_va)

            if balancer is not None:
                X_tr, y_tr = balancer.fit_resample(X_tr, y_tr)

            model = model_factory()
            model.fit(X_tr, y_tr)
            y_prob = model.predict_prob(X_va)
            y_pred = model.predict(X_va)
            fold_metrics.append(Metrics.evaluate_all(y_va, y_pred, y_prob))

        # Return mean and std for each key metric across folds
        summary = {}
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "cost"):
            vals = [m[key] for m in fold_metrics if key in m]
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))
        return summary