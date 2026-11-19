import numpy as np


class Metrics: # Evaluation metrics for binary classification

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        return tn, fp, fn, tp

    @staticmethod
    def accuracy(y_true, y_pred) -> float:
        tn, fp, fn, tp = Metrics.confusion_matrix(y_true, y_pred)
        total = tn + fp + fn + tp
        return (tp + tn) / total if total > 0 else 0.0

    @staticmethod
    def precision(y_true, y_pred) -> float:
        _, fp, _, tp = Metrics.confusion_matrix(y_true, y_pred)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def recall(y_true, y_pred) -> float:
        _, _, fn, tp = Metrics.confusion_matrix(y_true, y_pred)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def f1_score(y_true, y_pred) -> float:
        p = Metrics.precision(y_true, y_pred)
        r = Metrics.recall(y_true, y_pred)
        return (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

    @staticmethod
    def roc_auc(y_true, y_prob_pos) -> float:
        y_t = np.asarray(y_true).ravel()
        scores = np.asarray(y_prob_pos).ravel()
        # Sort by predicted probability descending
        order = np.argsort(scores)[::-1]
        y_sorted = y_t[order]

        n_pos = np.sum(y_sorted == 1)
        n_neg = np.sum(y_sorted == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.0

        tpr = np.cumsum(y_sorted == 1) / n_pos
        fpr = np.cumsum(y_sorted == 0) / n_neg
        # Prepend origin (0, 0) and compute area via trapezoidal rule
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        return float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2.0))

    @staticmethod
    def total_cost(y_true, y_pred, fp_cost: float = 2.0, fn_cost: float = 1.0) -> float:
        _, fp, fn, _ = Metrics.confusion_matrix(y_true, y_pred)
        return float(fp_cost * fp + fn_cost * fn)

    @staticmethod
    def evaluate_all(y_true, y_pred, y_prob=None, fp_cost: float = 2.0, fn_cost: float = 1.0) -> dict:
        tn, fp, fn, tp = Metrics.confusion_matrix(y_true, y_pred)
        results = {
            "accuracy": Metrics.accuracy(y_true, y_pred),
            "precision": Metrics.precision(y_true, y_pred),
            "recall": Metrics.recall(y_true, y_pred),
            "f1": Metrics.f1_score(y_true, y_pred),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "cost": Metrics.total_cost(y_true, y_pred, fp_cost, fn_cost),
        }
        if y_prob is not None:
            prob_pos = y_prob[:, 1] if np.ndim(y_prob) == 2 else y_prob
            results["roc_auc"] = Metrics.roc_auc(y_true, prob_pos)
        return results