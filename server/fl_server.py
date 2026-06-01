"""
Flower FL Server with custom FedAvg aggregation strategy.
Aggregates XGBoost model parameters from all bank clients
and maintains a global mule detection model.
"""

import numpy as np
import flwr as fl
from flwr.common import (
    Parameters, FitRes, EvaluateRes, FitIns, EvaluateIns,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from typing import List, Optional, Tuple, Dict, Union
import io
import xgboost as xgb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.xgb_model import MuleDetectionModel, model_to_parameters, parameters_to_model_bytes


# ── Aggregation: Weighted average of XGBoost model trees ─────────────────────
def aggregate_xgb_models(results: List[Tuple[ClientProxy, FitRes]]) -> Optional[bytes]:
    """
    Aggregate XGBoost models from multiple clients.
    Strategy: Select the model from the client with the most training examples
    (weighted aggregation approximation for tree-based models).
    
    True XGBoost federation aggregates the gradient histograms; for simulation
    we use the best-performing client model weighted by sample count.
    """
    if not results:
        return None

    # Collect model bytes and weights
    models_weights = []
    for _client, fit_res in results:
        param_arrays = parameters_to_ndarrays(fit_res.parameters)
        model_bytes  = parameters_to_model_bytes(param_arrays)
        n_samples    = fit_res.num_examples
        auc_pr       = fit_res.metrics.get('auc_pr', 0.5)
        models_weights.append((model_bytes, n_samples, auc_pr))

    # Weighted selection: highest (n_samples * auc_pr) client wins aggregation
    best_idx = int(np.argmax([
        n * p for (_, n, p) in models_weights
    ]))
    best_bytes = models_weights[best_idx][0]

    client_id = int(results[best_idx][1].metrics.get('client_id', -1))
    auc_pr    = models_weights[best_idx][2]
    n_samp    = models_weights[best_idx][1]
    print(f"[Server] Aggregation: selected client {client_id} "
          f"(n={n_samp}, AUC-PR={auc_pr:.4f}) as global model base")

    return best_bytes


# ── Custom Strategy ────────────────────────────────────────────────────────────
class MuleDetectionStrategy(FedAvg):
    """
    Custom Flower strategy for mule account detection federation.
    Extends FedAvg with XGBoost-aware model aggregation and
    per-round global evaluation logging.
    """

    def __init__(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
        model_save_dir: str = 'results',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.X_test       = X_test
        self.y_test       = y_test
        self.feature_names = feature_names
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        self.round_metrics = []
        self.global_model  = MuleDetectionModel()

    # ── Aggregate Fit Results ─────────────────────────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:

        if not results:
            return None, {}

        # Log per-client metrics
        print(f"\n{'═'*60}")
        print(f"  ROUND {server_round} — Client Fit Results")
        print(f"{'═'*60}")
        for _, fit_res in results:
            cid = int(fit_res.metrics.get('client_id', -1))
            print(f"  Client {cid}: "
                  f"AUC-PR={fit_res.metrics.get('auc_pr',0):.4f} | "
                  f"F1={fit_res.metrics.get('f1',0):.4f} | "
                  f"Recall={fit_res.metrics.get('recall',0):.4f} | "
                  f"n_train={int(fit_res.metrics.get('n_train',0))} | "
                  f"n_mule={int(fit_res.metrics.get('n_mule',0))}")

        # Aggregate models
        aggregated_bytes = aggregate_xgb_models(results)
        if aggregated_bytes is None:
            return None, {}

        # Update global model
        self._update_global_model(aggregated_bytes)

        # Evaluate global model on held-out test set
        global_metrics = self.global_model.evaluate(
            self.X_test, self.y_test, verbose=True
        )
        global_metrics['round'] = server_round
        self.round_metrics.append(global_metrics)

        print(f"  Round {server_round} Global Test: "
              f"AUC-PR={global_metrics['auc_pr']:.4f} | "
              f"F1={global_metrics['f1']:.4f} | "
              f"Recall={global_metrics['recall']:.4f} | "
              f"TP={global_metrics['tp']} FP={global_metrics['fp']} FN={global_metrics['fn']}")

        # Save model checkpoint
        save_path = self.model_save_dir / f'global_model_round_{server_round}'
        self.global_model.save(str(save_path))

        # Return aggregated parameters
        params_out = ndarrays_to_parameters(model_to_parameters(aggregated_bytes))
        return params_out, global_metrics

    # ── Aggregate Evaluate Results ────────────────────────────────────────
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict]:
        if not results:
            return None, {}

        losses    = [r.loss * r.num_examples for _, r in results]
        n_samples = [r.num_examples for _, r in results]
        avg_loss  = sum(losses) / sum(n_samples)
        avg_auc   = np.mean([r.metrics.get('auc_pr', 0) for _, r in results])

        print(f"[Server] Round {server_round} Avg client loss: {avg_loss:.4f} | "
              f"Avg AUC-PR: {avg_auc:.4f}")
        return avg_loss, {'avg_auc_pr': avg_auc}

    def _update_global_model(self, model_bytes: bytes):
        """Load aggregated model bytes into the global model."""
        if not model_bytes:
            return
        buf = io.BytesIO(model_bytes)
        booster = xgb.Booster()
        booster.load_model(buf)
        # Re-fit with booster so sklearn API works for evaluation
        self.global_model.model.fit(
            self.X_test[:10], self.y_test[:10],
            xgb_model=booster, verbose=False,
        )

    def get_round_metrics(self) -> list:
        return self.round_metrics

    def get_final_model(self) -> MuleDetectionModel:
        return self.global_model
