"""
Data loading and partitioning for Federated Learning.
Simulates a multi-bank federation — each bank is a separate Flower client
with its own local data shard.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.feature_engineering import engineer_features, get_feature_names


# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH   = Path(__file__).parent.parent / 'DataSet.csv'
NUM_CLIENTS    = 5        # Simulating 5 banks as FL clients
RANDOM_STATE   = 42
TEST_SIZE      = 0.20     # 20% held out globally for evaluation
VAL_SIZE       = 0.15     # 15% of training for local validation
SMOTE_STRATEGY = 0.3      # oversample minority to 30% of majority


# ── Load & Engineer ────────────────────────────────────────────────────────────
def load_and_engineer(csv_path: str = None) -> tuple:
    """
    Load raw CSV, apply feature engineering, return (X, y) arrays.
    """
    path = csv_path or DATASET_PATH
    print(f"[DataLoader] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[DataLoader] Raw shape: {df.shape}")

    engineered = engineer_features(df)
    feature_cols = get_feature_names()

    X = engineered[feature_cols].values.astype(np.float32)
    y = engineered['label'].values.astype(np.int32)

    print(f"[DataLoader] Engineered shape: {X.shape}")
    print(f"[DataLoader] Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, feature_cols


# ── Global Train/Test Split ────────────────────────────────────────────────────
def global_train_test_split(X, y):
    """Hold out a global test set that no client trains on."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[DataLoader] Global train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Federated Partition ────────────────────────────────────────────────────────
def partition_for_clients(X_train, y_train, num_clients: int = NUM_CLIENTS):
    """
    Partition training data across N clients (banks).
    Uses stratified partitioning to ensure each client has
    at least some positive (mule) examples.
    """
    client_data = {}

    # Separate majority / minority
    idx_maj = np.where(y_train == 0)[0]
    idx_min = np.where(y_train == 1)[0]

    np.random.seed(RANDOM_STATE)
    np.random.shuffle(idx_maj)
    np.random.shuffle(idx_min)

    # Split minority (mule accounts) across clients
    min_splits = np.array_split(idx_min, num_clients)
    maj_splits = np.array_split(idx_maj, num_clients)

    for cid in range(num_clients):
        idx = np.concatenate([maj_splits[cid], min_splits[cid]])
        np.random.shuffle(idx)

        X_c = X_train[idx]
        y_c = y_train[idx]

        # Local train / val split
        X_ctr, X_cval, y_ctr, y_cval = train_test_split(
            X_c, y_c, test_size=VAL_SIZE, stratify=y_c, random_state=RANDOM_STATE
        )

        client_data[cid] = {
            'X_train': X_ctr,
            'y_train': y_ctr,
            'X_val':   X_cval,
            'y_val':   y_cval,
            'n_samples': len(y_c),
            'n_positive': int(y_c.sum()),
        }
        print(f"[DataLoader] Client {cid}: {len(y_c)} samples | "
              f"{int(y_c.sum())} mule accounts | "
              f"train={len(y_ctr)}, val={len(y_cval)}")

    return client_data


# ── Scaling ────────────────────────────────────────────────────────────────────
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit scaler on global training data. Each client receives this scaler."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X).astype(np.float32)


# ── SMOTE for local oversampling ───────────────────────────────────────────────
def apply_smote(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Apply SMOTE locally on each client's training data.
    Only applies if there are enough minority samples (≥6).
    """
    n_pos = int(y_train.sum())
    if n_pos < 6:
        print(f"[SMOTE] Skipped — only {n_pos} positive samples (need ≥6)")
        return X_train, y_train

    k = min(5, n_pos - 1)
    smote = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        k_neighbors=k,
        random_state=RANDOM_STATE
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[SMOTE] {len(y_train)} → {len(y_res)} samples | "
          f"pos: {n_pos} → {int(y_res.sum())}")
    return X_res.astype(np.float32), y_res.astype(np.int32)


# ── Full Pipeline ──────────────────────────────────────────────────────────────
def prepare_federated_data(csv_path: str = None, num_clients: int = NUM_CLIENTS):
    """
    Full data preparation pipeline:
    1. Load + engineer features
    2. Global train/test split
    3. Fit scaler on training data
    4. Partition training data across N clients
    5. Scale each client's data
    Returns: client_data dict, X_test, y_test, scaler, feature_names
    """
    X, y, feature_names = load_and_engineer(csv_path)
    X_train, X_test, y_train, y_test = global_train_test_split(X, y)

    # Fit scaler once on global training data
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(scaler, X_train)
    X_test_scaled  = apply_scaler(scaler, X_test)

    # Partition among clients and apply SMOTE locally
    raw_client_data = partition_for_clients(X_train_scaled, y_train, num_clients)

    client_data = {}
    for cid, cdata in raw_client_data.items():
        X_res, y_res = apply_smote(cdata['X_train'], cdata['y_train'])
        client_data[cid] = {
            **cdata,
            'X_train': X_res,
            'y_train': y_res,
        }

    return client_data, X_test_scaled, y_test, scaler, feature_names


if __name__ == '__main__':
    client_data, X_test, y_test, scaler, feature_names = prepare_federated_data()
    print(f"\n[DataLoader] Feature names ({len(feature_names)}):")
    for i, fn in enumerate(feature_names):
        print(f"  {i:2d}. {fn}")
    print(f"\n[DataLoader] Global test set: {X_test.shape}, mule rate: {y_test.mean():.4f}")
