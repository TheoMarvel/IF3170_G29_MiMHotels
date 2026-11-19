import time
import pandas as pd
from src.evaluation.metrics import Metrics


class ModelComparator: # Compare scratch vs scikit-learn models side-by-side

    @staticmethod
    def compare(models: dict, X_train, y_train, X_val, y_val, fp_cost: float = 2.0, fn_cost: float = 1.0) -> pd.DataFrame:
        rows = []

        for name, model in models.items():
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            y_prob = model.predict_prob(X_val)
            y_pred = model.predict(X_val)
            pred_time = time.perf_counter() - t1

            scores = Metrics.evaluate_all(y_val, y_pred, y_prob, fp_cost=fp_cost, fn_cost=fn_cost)
            rows.append({
                "model": name,
                "accuracy": round(scores["accuracy"], 4),
                "precision": round(scores["precision"], 4),
                "recall": round(scores["recall"], 4),
                "f1": round(scores["f1"], 4),
                "roc_auc": round(scores.get("roc_auc", 0.0), 4),
                "cost": scores["cost"],
                "train_time_s": round(train_time, 4),
                "pred_time_s": round(pred_time, 4),
            })

        return pd.DataFrame(rows)