"""
Main Flower Federated Learning Simulation Runner
Mule Account Detection System

Run:
    python main_simulation.py --csv /path/to/DataSet.csv --rounds 10 --clients 5

Architecture:
    - 5 clients  → 5 banks (each with private transaction data)
    - 1 server   → coordinates aggregation (no raw data access)
    - XGBoost    → base model (tree boosting, handles imbalance)
    - SMOTE      → local oversampling to handle 0.89% mule rate
    - Flower     → FL framework for orchestration
"""

import argparse
import json
import time
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import flwr as fl

from data.data_loader import prepare_federated_data, NUM_CLIENTS
from clients.fl_client import make_client
from server.fl_server import MuleDetectionStrategy
from models.xgb_model import MuleDetectionModel, parameters_to_model_bytes


# ── CLI Arguments ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description='Mule Detection FL Simulation')
    parser.add_argument('--csv',     type=str,   default='DataSet.csv',
                        help='Path to DataSet.csv')
    parser.add_argument('--rounds',  type=int,   default=10,
                        help='Number of FL rounds')
    parser.add_argument('--clients', type=int,   default=NUM_CLIENTS,
                        help='Number of federated clients (banks)')
    parser.add_argument('--results', type=str,   default='results',
                        help='Directory to save models and metrics')
    parser.add_argument('--threshold', type=float, default=0.35,
                        help='Classification threshold (default 0.35 for high recall)')
    return parser.parse_args()


# ── Simulation Entry Point ─────────────────────────────────────────────────────
def run_simulation(args):
    print("\n" + "═"*65)
    print("  MULE ACCOUNT DETECTION — FEDERATED LEARNING SIMULATION")
    print("  Framework : Flower (flwr)")
    print("  Model     : XGBoost (tree boosting)")
    print("  Clients   : {} banks".format(args.clients))
    print("  Rounds    : {}".format(args.rounds))
    print("═"*65 + "\n")

    t0 = time.time()

    # ── Step 1: Prepare Data ─────────────────────────────────────────────
    print("[Step 1/4] Preparing federated data...")
    client_data, X_test, y_test, scaler, feature_names = prepare_federated_data(
        csv_path=args.csv,
        num_clients=args.clients,
    )
    print(f"  ✓ Data ready | Features: {len(feature_names)} | "
          f"Test mules: {int(y_test.sum())}/{len(y_test)}\n")

    # ── Step 2: Build Strategy ────────────────────────────────────────────
    print("[Step 2/4] Configuring FL strategy...")
    strategy = MuleDetectionStrategy(
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        model_save_dir=args.results,
        # FedAvg base params
        fraction_fit=1.0,            # all clients participate in fit
        fraction_evaluate=1.0,       # all clients participate in eval
        min_fit_clients=args.clients,
        min_evaluate_clients=args.clients,
        min_available_clients=args.clients,
    )
    print(f"  ✓ Strategy: MuleDetectionStrategy (FedAvg base)\n")

    # ── Step 3: Run Simulation ────────────────────────────────────────────
    print(f"[Step 3/4] Starting Flower simulation ({args.rounds} rounds)...")
    print(f"{'─'*65}")

    def client_fn(context) -> fl.client.Client:
        """Flower calls this to create each client instance."""
        # In flwr >= 1.0, context.node_config contains the node_id
        cid = int(context.node_config.get('partition-id', 0)) % args.clients
        return make_client(cid, client_data)

    history = fl.simulation.run_simulation(
        server_app=fl.server.ServerApp(
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
        ),
        client_app=fl.client.ClientApp(client_fn=client_fn),
        num_supernodes=args.clients,
        backend_config={"client_resources": {"num_cpus": 1}},
    )

    print(f"\n{'─'*65}")
    print("[Step 3/4] ✓ Simulation complete\n")

    # ── Step 4: Final Evaluation & Report ────────────────────────────────
    print("[Step 4/4] Final evaluation on held-out global test set...")
    final_model = strategy.get_final_model()

    print("\n  ── Final Global Model Performance ──")
    final_metrics = final_model.evaluate(
        X_test, y_test,
        threshold=args.threshold,
        verbose=True,
    )

    # Feature importance
    final_model.print_top_features(feature_names, top_n=15)

    # Round progression
    round_metrics = strategy.get_round_metrics()
    print("\n  ── Round-by-Round Progression ──")
    print(f"  {'Round':<8} {'AUC-PR':<10} {'AUC-ROC':<10} {'F1':<8} {'Recall':<10} {'Precision':<12} {'TP':<5} {'FP':<5} {'FN'}")
    for rm in round_metrics:
        print(f"  {rm['round']:<8} {rm['auc_pr']:<10.4f} {rm['auc_roc']:<10.4f} "
              f"{rm['f1']:<8.4f} {rm['recall']:<10.4f} {rm['precision']:<12.4f} "
              f"{rm['tp']:<5} {rm['fp']:<5} {rm['fn']}")

    # Save all metrics to JSON
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'config': vars(args),
        'final_metrics': final_metrics,
        'round_metrics': round_metrics,
        'features': feature_names,
        'n_features': len(feature_names),
        'n_clients': args.clients,
        'n_rounds': args.rounds,
        'elapsed_seconds': round(time.time() - t0, 1),
    }
    report_path = results_dir / 'fl_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  ✓ Report saved to {report_path}")

    # Save final model
    final_model.save(str(results_dir / 'final_global_model'))

    print(f"\n{'═'*65}")
    print(f"  Total time : {time.time()-t0:.1f}s")
    print(f"  AUC-PR     : {final_metrics['auc_pr']:.4f}")
    print(f"  F1 Score   : {final_metrics['f1']:.4f}")
    print(f"  Recall     : {final_metrics['recall']:.4f}")
    print(f"  Precision  : {final_metrics['precision']:.4f}")
    print(f"  TP (Caught): {final_metrics['tp']} mule accounts")
    print(f"  FN (Missed): {final_metrics['fn']} mule accounts")
    print(f"  FP (False alarm): {final_metrics['fp']}")
    print("═"*65 + "\n")

    return report


if __name__ == '__main__':
    args = parse_args()
    run_simulation(args)
