"""
Flower FL Client — represents one bank in the federation.
Each client:
  1. Receives global model parameters from server
  2. Trains locally on its private data shard
  3. Sends updated model back to server
  4. Never shares raw transaction data
"""

import numpy as np
import flwr as fl
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    GetParametersIns, GetParametersRes, Status, Code, ndarrays_to_parameters,
    parameters_to_ndarrays
)
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgb_model import MuleDetectionModel, model_to_parameters, parameters_to_model_bytes
import xgboost as xgb


class MuleBankClient(fl.client.Client):
    """
    Flower client representing one bank's federated learning node.

    Privacy: Only model parameters (gradient boosting tree structure)
    are shared — never raw customer transaction data.
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        self.client_id = client_id
        self.X_train   = X_train
        self.y_train   = y_train
        self.X_val     = X_val
        self.y_val     = y_val
        self.model     = MuleDetectionModel()
        self._local_booster = None   # tracks accumulated booster across rounds
        self._round = 0

        print(f"[Client {self.client_id}] Initialized | "
              f"train={len(y_train)} val={len(y_val)} | "
              f"pos_rate={y_train.mean():.3f}")

    # ── Get Parameters (server asks for current model state) ──────────────
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        model_bytes = self.model.get_xgb_model_bytes()
        params      = ndarrays_to_parameters(model_to_parameters(model_bytes))
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=params,
        )

    # ── Fit (local training round) ─────────────────────────────────────────
    def fit(self, ins: FitIns) -> FitRes:
        self._round += 1
        print(f"\n[Client {self.client_id}] ── Round {self._round} ──────────────")

        # 1. Receive global model from server
        param_arrays = parameters_to_ndarrays(ins.parameters)
        model_bytes  = parameters_to_model_bytes(param_arrays)

        # 2. Load global model state (if any)
        if model_bytes:
            buf = __import__('io').BytesIO(model_bytes)
            self._local_booster = xgb.Booster()
            self._local_booster.load_model(buf)
            print(f"[Client {self.client_id}] Loaded global model ({len(model_bytes)} bytes)")
        else:
            self._local_booster = None
            print(f"[Client {self.client_id}] Starting fresh (round 1)")

        # 3. Local training on private data
        self.model.fit(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            xgb_model=self._local_booster,
        )

        # 4. Local evaluation
        metrics = self.model.evaluate(self.X_val, self.y_val, verbose=False)
        print(f"[Client {self.client_id}] Val  AUC-PR={metrics['auc_pr']:.4f} | "
              f"F1={metrics['f1']:.4f} | "
              f"Recall={metrics['recall']:.4f} | "
              f"Precision={metrics['precision']:.4f}")

        # 5. Serialize updated model to send back
        model_bytes_out = self.model.get_xgb_model_bytes()
        params_out = ndarrays_to_parameters(model_to_parameters(model_bytes_out))

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=params_out,
            num_examples=len(self.y_train),
            metrics={
                'client_id': float(self.client_id),
                'auc_pr':    metrics['auc_pr'],
                'f1':        metrics['f1'],
                'recall':    metrics['recall'],
                'precision': metrics['precision'],
                'auc_roc':   metrics['auc_roc'],
                'n_train':   float(len(self.y_train)),
                'n_mule':    float(int(self.y_train.sum())),
            },
        )

    # ── Evaluate (server asks client to evaluate current global model) ────
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model for evaluation
        param_arrays = parameters_to_ndarrays(ins.parameters)
        model_bytes  = parameters_to_model_bytes(param_arrays)

        if model_bytes:
            buf = __import__('io').BytesIO(model_bytes)
            booster = xgb.Booster()
            booster.load_model(buf)
            # Temporary evaluation model
            eval_model = MuleDetectionModel()
            eval_model._local_booster = booster
            eval_model.model.fit(
                self.X_val, self.y_val,
                xgb_model=booster,
                verbose=False,
            )
            metrics = eval_model.evaluate(self.X_val, self.y_val, verbose=False)
        else:
            metrics = {'auc_pr': 0.0, 'f1': 0.0, 'recall': 0.0, 'auc_roc': 0.5}

        # Flower expects a scalar 'loss'
        loss = 1.0 - metrics['auc_pr']

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(loss),
            num_examples=len(self.y_val),
            metrics={
                'auc_pr':    metrics['auc_pr'],
                'f1':        metrics['f1'],
                'recall':    metrics['recall'],
                'auc_roc':   metrics['auc_roc'],
                'client_id': float(self.client_id),
            },
        )


# ── Client Factory ────────────────────────────────────────────────────────────
def make_client(client_id: int, client_data: dict) -> MuleBankClient:
    """Factory function used by the simulation runner."""
    cdata = client_data[client_id]
    return MuleBankClient(
        client_id=client_id,
        X_train=cdata['X_train'],
        y_train=cdata['y_train'],
        X_val=cdata['X_val'],
        y_val=cdata['y_val'],
    )
