"""
XGBoost Model for Mule Account Detection.
Wrapped to be compatible with Flower FL parameter exchange
(serialize/deserialize model parameters as numpy arrays).
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, f1_score,
    confusion_matrix, average_precision_score
)
import json
import io
import pickle
from pathlib import Path


# ── Model Hyperparameters ──────────────────────────────────────────────────────
XGB_PARAMS = {
    'objective':        'binary:logistic',
    'eval_metric':      ['logloss', 'aucpr'],
    'tree_method':      'hist',
    'max_depth':        6,
    'learning_rate':    0.05,
    'n_estimators':     100,          # trees per FL round
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma':            1,
    'reg_alpha':        0.5,
    'reg_lambda':       2.0,
    'scale_pos_weight': 50,           # handles 1:111 class imbalance (~111 negatives per positive)
    'random_state':     42,
    'n_jobs':           -1,
    'use_label_encoder': False,
    'verbosity':        0,
}

# Trees added per FL round (incremental training)
TREES_PER_ROUND = 10

# Threshold for converting probability to label (tuned for high recall)
CLASSIFICATION_THRESHOLD = 0.35


# ── Model Class ───────────────────────────────────────────────────────────────
class MuleDetectionModel:
    """
    XGBoost classifier for mule account detection.
    Supports incremental training for Flower federated rounds.
    """

    def __init__(self, params: dict = None):
        self.params = params or XGB_PARAMS
        self.model: xgb.XGBClassifier = None
        self.n_trees_total = 0
        self._init_model()

    def _init_model(self):
        self.model = xgb.XGBClassifier(**{
            **self.params,
            'n_estimators': TREES_PER_ROUND,
        })

    # ── Training ──────────────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            xgb_model=None):
        """
        Train the model. If xgb_model is provided, continues training
        from that checkpoint (incremental learning for FL rounds).
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
            xgb_model=xgb_model,
        )
        self.n_trees_total += TREES_PER_ROUND
        return self

    # ── Inference ─────────────────────────────────────────────────────────
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = CLASSIFICATION_THRESHOLD) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y: np.ndarray,
                 threshold: float = CLASSIFICATION_THRESHOLD,
                 verbose: bool = True) -> dict:
        proba = self.predict_proba(X)
        preds = (proba >= threshold).astype(int)

        auc_roc = roc_auc_score(y, proba)
        auc_pr  = average_precision_score(y, proba)
        f1      = f1_score(y, preds, zero_division=0)
        cm      = confusion_matrix(y, preds)

        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)

        metrics = {
            'auc_roc':   round(auc_roc, 4),
            'auc_pr':    round(auc_pr, 4),
            'f1':        round(f1, 4),
            'precision': round(precision, 4),
            'recall':    round(recall, 4),
            'tp': int(tp), 'fp': int(fp),
            'tn': int(tn), 'fn': int(fn),
            'threshold': threshold,
        }

        if verbose:
            print(f"\n{'─'*50}")
            print(f"  AUC-ROC : {auc_roc:.4f}")
            print(f"  AUC-PR  : {auc_pr:.4f}  (primary metric for imbalanced data)")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}  |  Recall: {recall:.4f}")
            print(f"  Confusion Matrix:")
            print(f"    TN={tn}  FP={fp}")
            print(f"    FN={fn}  TP={tp}")
            print(f"{'─'*50}\n")

        return metrics

    # ── Parameter Serialization (for Flower FL) ───────────────────────────
    def get_xgb_model_bytes(self) -> bytes:
        """Serialize XGBoost model to bytes for Flower parameter exchange."""
        if self.model is None:
            return b''
        buf = io.BytesIO()
        self.model.get_booster().save_model(buf)
        return buf.getvalue()

    def set_xgb_model_bytes(self, model_bytes: bytes):
        """Deserialize XGBoost model from bytes received from Flower server."""
        if not model_bytes:
            return
        buf = io.BytesIO(model_bytes)
        booster = xgb.Booster()
        booster.load_model(buf)
        # Wrap back into sklearn API
        self._init_model()
        self.model.get_booster = lambda: booster
        self.model._Booster = booster

    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path.with_suffix('.json')))
        print(f"[Model] Saved to {path.with_suffix('.json')}")

    def load(self, path: str):
        """Load model from disk."""
        self._init_model()
        self.model.load_model(str(Path(path).with_suffix('.json')))
        print(f"[Model] Loaded from {path}")
        return self

    # ── Feature Importance ────────────────────────────────────────────────
    def get_feature_importance(self, feature_names: list) -> dict:
        """Returns sorted feature importances."""
        imp = self.model.feature_importances_
        return dict(sorted(
            zip(feature_names, imp.tolist()),
            key=lambda x: x[1], reverse=True
        ))

    def print_top_features(self, feature_names: list, top_n: int = 15):
        imp_dict = self.get_feature_importance(feature_names)
        print(f"\n{'─'*40}")
        print(f"  Top {top_n} Features by Importance:")
        for i, (name, score) in enumerate(list(imp_dict.items())[:top_n], 1):
            bar = '█' * int(score * 300)
            print(f"  {i:2d}. {name:<32} {score:.4f}  {bar}")
        print(f"{'─'*40}\n")


# ── Flower-Compatible Parameter Functions ─────────────────────────────────────
def model_to_parameters(model_bytes: bytes) -> list:
    """
    Convert model bytes to list of numpy arrays
    (Flower expects List[np.ndarray] as parameters).
    """
    if not model_bytes:
        return [np.array([], dtype=np.uint8)]
    arr = np.frombuffer(model_bytes, dtype=np.uint8)
    return [arr]


def parameters_to_model_bytes(parameters: list) -> bytes:
    """Convert Flower parameters back to model bytes."""
    if not parameters or len(parameters[0]) == 0:
        return b''
    return parameters[0].tobytes()
