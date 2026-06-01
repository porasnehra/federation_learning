"""
Real-time Mule Account Risk Scorer
Loads the trained global model and scores new account/transaction records.
Outputs a risk level + triggered signals for each account.

Usage:
    python predict.py --model results/final_global_model --input new_accounts.csv
    python predict.py --model results/final_global_model --demo
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from models.xgb_model import MuleDetectionModel, CLASSIFICATION_THRESHOLD
from utils.feature_engineering import engineer_features, get_feature_names

# Risk tier thresholds
RISK_TIERS = {
    'CRITICAL': 0.80,
    'HIGH':     0.50,
    'MEDIUM':   0.35,
    'LOW':      0.0,
}

SIGNAL_NAMES = {
    'is_high_value_in':       "Signal 1: High-Value Inward Transfer",
    'high_tx_count_flag':     "Signal 2: Rapid Transaction Velocity (Smurfing)",
    'urban_high_volume':      "Signal 3: Geographical High-Volume Anomaly",
    'bpl_high_inflow':        "Signal 4: BPL/Low-Credit + High Inflow",
    'age_outlier':            "Signal 5: Age Outlier",
    'student_high_velocity':  "Signal 6: Student Account High Velocity",
    'student_high_tx':        "Signal 6b: Student Account High Tx Count",
    'multi_linked_flag':      "Signal 7: Multiple Linked Accounts",
    'velocity_spike':         "Signal 8: Historical Baseline Deviation",
    'rare_auth_flag':         "Signal 9: Linked Device Auth Flag",
    'device_batch_flag':      "Signal 10: Device Fingerprint / Batch Opening",
    'is_new_account':         "Signal 11: New Account (SIM/Telecom Age)",
    'severe_event_flag':      "Severe Risk Event (F3923)",
    'passthrough_ratio':      "Pass-Through Ratio Near 0.5 (Fund Circulation)",
}


def get_risk_tier(prob: float) -> str:
    for tier, threshold in RISK_TIERS.items():
        if prob >= threshold:
            return tier
    return 'LOW'


def explain_signals(row: pd.Series) -> list:
    """Return list of triggered fraud signals for a given record."""
    triggered = []
    feat = row.to_dict()

    if feat.get('is_high_value_in', 0) == 1:
        triggered.append(SIGNAL_NAMES['is_high_value_in'])
    if feat.get('high_tx_count_flag', 0) == 1:
        triggered.append(SIGNAL_NAMES['high_tx_count_flag'])
    if feat.get('urban_high_volume', 0) == 1:
        triggered.append(SIGNAL_NAMES['urban_high_volume'])
    if feat.get('bpl_high_inflow', 0) == 1:
        triggered.append(SIGNAL_NAMES['bpl_high_inflow'])
    if feat.get('age_outlier', 0) == 1:
        triggered.append(SIGNAL_NAMES['age_outlier'])
    if feat.get('student_high_velocity', 0) == 1:
        triggered.append(SIGNAL_NAMES['student_high_velocity'])
    if feat.get('student_high_tx', 0) == 1:
        triggered.append(SIGNAL_NAMES['student_high_tx'])
    if feat.get('multi_linked_flag', 0) == 1:
        triggered.append(SIGNAL_NAMES['multi_linked_flag'])
    if feat.get('velocity_spike', 0) == 1:
        triggered.append(SIGNAL_NAMES['velocity_spike'])
    if feat.get('rare_auth_flag', 0) == 1:
        triggered.append(SIGNAL_NAMES['rare_auth_flag'])
    if feat.get('device_batch_flag', 0) == 1:
        triggered.append(SIGNAL_NAMES['device_batch_flag'])
    if feat.get('is_new_account', 0) == 1:
        triggered.append(SIGNAL_NAMES['is_new_account'])
    if feat.get('severe_event_flag', 0) >= 1:
        triggered.append(SIGNAL_NAMES['severe_event_flag'])
    if 0.40 < feat.get('passthrough_ratio', 0) < 0.60:
        triggered.append(SIGNAL_NAMES['passthrough_ratio'])

    return triggered


def score_dataframe(df: pd.DataFrame, model: MuleDetectionModel,
                    threshold: float = CLASSIFICATION_THRESHOLD) -> pd.DataFrame:
    """Score a raw DataFrame and return risk results."""
    features_df = engineer_features(df)
    feature_cols = get_feature_names()
    X = features_df[feature_cols].values.astype(np.float32)

    probas = model.predict_proba(X)
    preds  = (probas >= threshold).astype(int)

    results = []
    for i, (prob, pred) in enumerate(zip(probas, preds)):
        row = features_df.iloc[i]
        tier = get_risk_tier(prob)
        signals = explain_signals(row)
        results.append({
            'account_idx':    i,
            'mule_probability': round(float(prob), 4),
            'risk_tier':      tier,
            'flagged':        bool(pred),
            'signals_triggered': len(signals),
            'signals':        signals,
        })

    return pd.DataFrame(results)


def print_risk_report(results_df: pd.DataFrame, top_n: int = 20):
    """Print a human-readable risk assessment report."""
    print("\n" + "═"*70)
    print("  MULE ACCOUNT RISK ASSESSMENT REPORT")
    print("═"*70)

    # Summary
    n_total    = len(results_df)
    n_flagged  = results_df['flagged'].sum()
    n_critical = (results_df['risk_tier'] == 'CRITICAL').sum()
    n_high     = (results_df['risk_tier'] == 'HIGH').sum()
    n_medium   = (results_df['risk_tier'] == 'MEDIUM').sum()

    print(f"\n  Total accounts scored : {n_total}")
    print(f"  Flagged (≥threshold)  : {n_flagged} ({n_flagged/n_total*100:.2f}%)")
    print(f"  CRITICAL risk         : {n_critical}")
    print(f"  HIGH risk             : {n_high}")
    print(f"  MEDIUM risk           : {n_medium}")

    # Top N riskiest accounts
    top = results_df.nlargest(top_n, 'mule_probability')
    print(f"\n  Top {top_n} Highest-Risk Accounts:")
    print(f"  {'Idx':<8} {'Probability':<14} {'Risk Tier':<12} {'Signals':<10} {'Key Signals'}")
    print(f"  {'─'*8} {'─'*14} {'─'*12} {'─'*10} {'─'*40}")

    for _, row in top.iterrows():
        sigs = '; '.join(row['signals'][:2]) if row['signals'] else '—'
        tier_icon = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢'}.get(row['risk_tier'],'⚪')
        print(f"  {int(row['account_idx']):<8} {row['mule_probability']:<14.4f} "
              f"{tier_icon} {row['risk_tier']:<10} {row['signals_triggered']:<10} {sigs}")

    print(f"\n{'═'*70}\n")


def demo_mode():
    """Run a quick demo with synthetic edge cases."""
    print("\n[Predictor] Running DEMO mode with synthetic edge-case accounts...\n")
    demo_accounts = pd.DataFrame({
        # Columns from the dataset used by engineer_features
        'F3799': [6e7, 2e5, 1.4e8, 5e4, 2.36e9],     # total_in (INR)
        'F3800': [3e7, 9e4, 7e7, 2e4, 1.3e9],          # total_out
        'F3796': [699, 15, 331, 5,  754],               # tx_credit_count
        'F3797': [14,  8,  5,  3,  460],                # tx_debit_count
        'F3920': [0,   0,  0,  1,  0],                  # alerts
        'F3923': [0,   0,  0,  0,  0],                  # severe_event
        'F3919': [1,   1,  1,  1,  3],                  # linked_accounts
        'F3922': [0,   1,  0,  0,  2],                  # high_val_events
        'F3891': ['student','salaried','student','housewife','salaried'],
        'F3886': ['Savings','Savings','Savings','Savings','MSME Medium'],
        'F3889': ['G365D','G365D','G365D','G365D','G365D'],
        'F3890': ['SU','R','M','R','M'],
        'F3893': ['RETAIL','RETAIL','RETAIL','RETAIL','CORPORATE'],
        'F3894': [32., 35., 24., 47., 17.],             # age
        'F3895': [600, 600, 600, 600, 600],             # credit_score
        'F3887': [170, 168, 169, 166, 167],             # tenure_months
        'F3900': [0, 0, 0, 0, 0],                       # npa_flag
        'F3901': [0, 0, 0, 0, 0],
        'F3902': [0, 0, 0, 0, 0],
        'F3905': [0, 0, 0, 0, 0],
        'F3912': [0, 0, 0, 0, 0],
        'F3913': [1, 1, 1, 0, 1],
        'F3915': [0, 0, 0, 0, 0],
        'F3916': [1, 1, 1, 1, 1],
        'F13': [0.8, 0.5, 0.9, 0.4, 0.7],
        'F14': [0.86, 0.54, 0.9, 0.4, 0.75],
        'F15': [0.79, 0.46, 0.85, 0.38, 0.7],
        'F16': [0.95, 0.39, 0.99, 0.35, 0.85],
        'F19': [0.8, 0.5, 0.76, 0.46, 0.7],
        'F25': [0.8, 0.5, 0.76, 0.46, 0.7],
        'F3856': [1.2, 1.0, 1.8, 1.0, 1.1],
        'F3859': [1.1, 1.0, 1.7, 1.0, 1.0],
        'F3882': [0.5, -0.1, 1.2, -0.2, 0.3],
        'F3883': [-0.4, -0.5, -0.3, -0.8, -0.4],
        'F2796': [-0.8, -1.0, -0.7, -1.0, -0.9],
        'F3877': [-1.0, -1.0, -0.9, -1.0, -1.0],
    })

    labels = ['Row75-like (mule?)','Normal salaried','Row76-like (mule?)','Housewife low','MSME high-vol']
    return demo_accounts, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mule Account Risk Scorer')
    parser.add_argument('--model',    type=str, default='results/final_global_model.json')
    parser.add_argument('--input',    type=str, default=None, help='Input CSV to score')
    parser.add_argument('--demo',     action='store_true', help='Run demo with synthetic data')
    parser.add_argument('--threshold',type=float, default=CLASSIFICATION_THRESHOLD)
    parser.add_argument('--top_n',   type=int,  default=20)
    args = parser.parse_args()

    # Load model
    model = MuleDetectionModel()
    model.load(args.model)
    print(f"[Predictor] Model loaded from {args.model}")

    if args.demo:
        df, labels = demo_mode()
        features_df = engineer_features(df)
        X = features_df[get_feature_names()].values.astype(np.float32)
        probas = model.predict_proba(X)
        print(f"\n  {'Account':<30} {'Mule Prob':<12} {'Risk Tier'}")
        print(f"  {'─'*30} {'─'*12} {'─'*12}")
        for label, prob in zip(labels, probas):
            tier = get_risk_tier(prob)
            icon = {'CRITICAL':'🔴','HIGH':'🟠','MEDIUM':'🟡','LOW':'🟢'}.get(tier,'⚪')
            print(f"  {label:<30} {prob:<12.4f} {icon} {tier}")

    elif args.input:
        df = pd.read_csv(args.input)
        print(f"[Predictor] Scoring {len(df)} accounts from {args.input}")
        results = score_dataframe(df, model, threshold=args.threshold)
        print_risk_report(results, top_n=args.top_n)
        out = args.input.replace('.csv', '_risk_scores.csv')
        results.to_csv(out, index=False)
        print(f"[Predictor] Scores saved to {out}")
    else:
        print("Use --demo for a quick test or --input <csv> to score real data.")
