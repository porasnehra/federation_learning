"""
Feature Engineering for Mule Account Detection
Maps the 11 fraud detection signals to actual dataset columns.

Signal → Column Mapping:
 1. High-Value Transfers      → F3799 (total inward INR), F3800 (total outward INR), F3920 (alerts)
 2. Smurfing (rapid small tx) → F3796 (credit count), F3797 (debit count), F3922 (high-val events)
 3. Geographical Anomalies    → F3890 (zone: R/SU/U/M), encoded
 4. Income-to-Transfer (BPL)  → F3895 (credit score ≤400 = BPL proxy), F3894 (age), F3891 (occupation)
 5. Age Outliers              → F3894 (age), combined with transaction volume
 6. Student Account Velocity  → F3891 (occupation=student), F3796, F3797, F3799
 7. Network Density           → F3919 (linked accounts count)
 8. Historical Baselines      → F13–F18 (multi-window risk scores), F3856–F3858 (velocity)
 9. Linked Device Auth        → F3905 (rare auth flag), F3912 (device-linked flag)
10. Device Fingerprinting     → F3912, F3915 (rare flags correlated with batch account opening)
11. SIM & Telecom Age         → F3889 (tenure bucket: L7D/L14D/L30D = new SIM proxy)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ── Tenure bucket encoding (Signal 11: SIM age proxy) ──────────────────────
TENURE_RISK = {
    'L7D':   1.0,   # <7 days   → highest risk (new SIM / new account)
    'L14D':  0.85,
    'L30D':  0.70,
    'L90D':  0.40,
    'L180D': 0.20,
    'L365D': 0.10,
    'G365D': 0.0,   # >365 days → lowest risk
}

ZONE_ENC = {'R': 0, 'SU': 1, 'U': 2, 'M': 3}

OCCUPATION_STUDENT   = 'student'
OCCUPATION_RISK = {
    'student':      0.9,
    'housewife':    0.6,
    'agriculture':  0.3,
    'selfemployed': 0.5,
    'salaried':     0.2,
    'retired':      0.2,
    'others':       0.4,
}

SEGMENT_ENC = {'RETAIL': 0, 'CORPORATE': 1}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw DataFrame and returns an engineered feature matrix
    aligned to the 11 fraud signals. Also returns the target column.
    """
    out = pd.DataFrame(index=df.index)

    # ── 1. High-Value Transfer Signal ─────────────────────────────────────
    total_in  = df['F3799'].fillna(0)
    total_out = df['F3800'].fillna(0)
    tx_in_cnt  = df['F3796'].fillna(0)
    tx_out_cnt = df['F3797'].fillna(0)

    # log-scale amounts (handle zeros)
    out['log_total_in']  = np.log1p(total_in)
    out['log_total_out'] = np.log1p(total_out)

    # pass-through ratio: near 0.5 = balanced in/out = mule signature
    safe_total = total_in + total_out + 1e-9
    out['passthrough_ratio'] = total_out / safe_total   # ~0.5 → suspicious

    # large absolute transfer flag (above 99th percentile)
    p99 = total_in.quantile(0.99)
    out['is_high_value_in'] = (total_in > p99).astype(int)

    # ── 2. Smurfing / Rapid Small Transactions ────────────────────────────
    # many credits but each small → avg credit size
    out['avg_credit_size'] = total_in / (tx_in_cnt + 1)
    out['avg_debit_size']  = total_out / (tx_out_cnt + 1)

    # high tx count flag (smurfing: many small transactions)
    p95_cnt = tx_in_cnt.quantile(0.95)
    out['high_tx_count_flag'] = (tx_in_cnt > p95_cnt).astype(int)

    # credit/debit count imbalance (many credits, few debits)
    out['credit_debit_ratio'] = tx_in_cnt / (tx_out_cnt + 1)

    # raw counts
    out['tx_credit_count'] = tx_in_cnt
    out['tx_debit_count']  = tx_out_cnt

    # high-value event count (F3922)
    out['high_val_event_count'] = df['F3922'].fillna(0)

    # ── 3. Geographical Anomaly ───────────────────────────────────────────
    out['zone_risk'] = df['F3890'].map(ZONE_ENC).fillna(1).astype(int)
    # metro/urban accounts with high volume = higher risk
    out['urban_high_volume'] = (
        (out['zone_risk'] >= 2) & (total_in > total_in.quantile(0.75))
    ).astype(int)

    # ── 4. Income-to-Transfer / BPL Factor ───────────────────────────────
    credit_score = df['F3895'].fillna(600)
    out['credit_score']   = credit_score
    out['is_bpl_proxy']   = (credit_score <= 400).astype(int)   # poor credit = BPL proxy
    # BPL + high inflow = extreme red flag
    out['bpl_high_inflow'] = (
        (credit_score <= 400) & (total_in > total_in.quantile(0.80))
    ).astype(int)

    # ── 5. Age Outlier Signal ─────────────────────────────────────────────
    age = df['F3894'].fillna(35)
    out['age'] = age
    out['age_outlier'] = (
        (age < 18) | (age > 80)
    ).astype(int)
    # young age + high transaction volume
    out['young_high_volume'] = (
        (age < 25) & (total_in > total_in.quantile(0.70))
    ).astype(int)

    # ── 6. Student Account Velocity ───────────────────────────────────────
    occupation = df['F3891'].fillna('others').str.lower()
    out['occupation_risk'] = occupation.map(OCCUPATION_RISK).fillna(0.4)
    out['is_student']      = (occupation == OCCUPATION_STUDENT).astype(int)
    # student + high velocity = strongest mule signal in dataset
    student_velocity_threshold = total_in[occupation == OCCUPATION_STUDENT].quantile(0.90)
    out['student_high_velocity'] = (
        (occupation == OCCUPATION_STUDENT) &
        (total_in > student_velocity_threshold)
    ).astype(int)
    # student with high credit count
    out['student_high_tx'] = (
        (occupation == OCCUPATION_STUDENT) & (tx_in_cnt > 200)
    ).astype(int)

    # ── 7. Network Density ────────────────────────────────────────────────
    linked_accts = df['F3919'].fillna(1)
    out['linked_account_count'] = linked_accts
    out['multi_linked_flag']    = (linked_accts > 2).astype(int)

    # ── 8. Historical Transaction Baselines ───────────────────────────────
    # F13–F18: multi-window risk score triplets (Window 1 = 7-day, etc.)
    out['risk_score_w1_a'] = df['F13'].fillna(df['F13'].median())
    out['risk_score_w1_b'] = df['F14'].fillna(df['F14'].median())
    out['risk_score_w1_c'] = df['F15'].fillna(df['F15'].median())
    out['risk_score_w1_fine_a'] = df['F16'].fillna(df['F16'].median())

    # Window 2 (30-day)
    out['risk_score_w2_a'] = df['F19'].fillna(df['F19'].median())
    out['risk_score_w3_a'] = df['F25'].fillna(df['F25'].median())

    # Velocity features (F3856–F3861: current vs prior period ratio)
    out['velocity_ratio_1'] = df['F3856'].fillna(1.0)
    out['velocity_ratio_2'] = df['F3859'].fillna(1.0)
    # velocity > 1.5 = current period 50% above last period
    out['velocity_spike'] = ((df['F3856'].fillna(1.0) > 1.5) |
                              (df['F3859'].fillna(1.0) > 1.5)).astype(int)

    # Historical deviation from baseline (F3882–F3883: z-score type)
    out['baseline_deviation_1'] = df['F3882'].fillna(0)
    out['baseline_deviation_2'] = df['F3883'].fillna(0)

    # ── 9. Linked Device Authentication ──────────────────────────────────
    out['rare_auth_flag']    = df['F3905'].fillna(0)  # 0.1% positive, high signal
    out['npa_flag']          = df['F3900'].fillna(0)
    out['alert_count']       = df['F3920'].fillna(0)
    out['severe_event_flag'] = df['F3923'].fillna(0)

    # ── 10. Device Fingerprinting / Batch Account Opening ─────────────────
    out['device_batch_flag'] = df['F3915'].fillna(0)  # 0.3% rare batch flag
    out['account_flag_1']    = df['F3912'].fillna(0)  # 0.9% correlated with device
    out['account_flag_2']    = df['F3901'].fillna(0)  # STR-filed flag
    out['account_flag_3']    = df['F3902'].fillna(0)
    # Combined device fingerprint risk score
    out['device_risk_score'] = (
        out['rare_auth_flag'] * 3 +
        out['device_batch_flag'] * 2 +
        out['account_flag_1'] +
        out['account_flag_2']
    )

    # ── 11. SIM & Telecom Age ─────────────────────────────────────────────
    out['tenure_risk'] = df['F3889'].map(TENURE_RISK).fillna(0.0)
    out['is_new_account'] = (df['F3889'].isin(['L7D', 'L14D', 'L30D'])).astype(int)
    # New account + high velocity = very high risk
    out['new_acct_high_volume'] = (
        out['is_new_account'] & (total_in > total_in.quantile(0.70))
    ).astype(int)

    # ── Derived Composite Signals ──────────────────────────────────────────
    # Net position: near-zero = pass-through mule
    out['net_position'] = total_in - total_out
    out['net_position_ratio'] = out['net_position'] / (total_in + 1e-9)

    # Account type encoding
    acct_type_map = {
        'Savings': 0, 'Current': 1, 'MSME Micro': 2, 'MSME Small': 3,
        'MSME Medium': 4, 'Corp Adv': 5, 'Agri Adv': 6, 'Staff Loans': 7,
        'Term Deposit': 8, 'Gold Loan': 9, 'Others': 10
    }
    out['account_type'] = df['F3886'].map(acct_type_map).fillna(0).astype(int)

    # Segment
    out['segment'] = df['F3893'].map(SEGMENT_ENC).fillna(0).astype(int)

    # Account tenure in months
    out['tenure_months'] = df['F3887'].fillna(df['F3887'].median())

    # Anomaly score features (isolation forest outputs — negative=normal)
    out['anomaly_score_1'] = df['F2796'].fillna(-1.0)
    out['anomaly_score_2'] = df['F3877'].fillna(-1.0)

    # ── Target ─────────────────────────────────────────────────────────────
    if 'F3924' in df.columns:
        out['label'] = df['F3924'].fillna(0).astype(int)

    return out


def get_feature_names() -> list:
    """Returns ordered list of all engineered feature column names."""
    return [
        # Signal 1: High-Value Transfers
        'log_total_in', 'log_total_out', 'passthrough_ratio', 'is_high_value_in',
        # Signal 2: Smurfing
        'avg_credit_size', 'avg_debit_size', 'high_tx_count_flag',
        'credit_debit_ratio', 'tx_credit_count', 'tx_debit_count', 'high_val_event_count',
        # Signal 3: Geography
        'zone_risk', 'urban_high_volume',
        # Signal 4: BPL
        'credit_score', 'is_bpl_proxy', 'bpl_high_inflow',
        # Signal 5: Age
        'age', 'age_outlier', 'young_high_volume',
        # Signal 6: Student Velocity
        'occupation_risk', 'is_student', 'student_high_velocity', 'student_high_tx',
        # Signal 7: Network Density
        'linked_account_count', 'multi_linked_flag',
        # Signal 8: Historical Baselines
        'risk_score_w1_a', 'risk_score_w1_b', 'risk_score_w1_c', 'risk_score_w1_fine_a',
        'risk_score_w2_a', 'risk_score_w3_a',
        'velocity_ratio_1', 'velocity_ratio_2', 'velocity_spike',
        'baseline_deviation_1', 'baseline_deviation_2',
        # Signal 9: Device Auth
        'rare_auth_flag', 'npa_flag', 'alert_count', 'severe_event_flag',
        # Signal 10: Device Fingerprinting
        'device_batch_flag', 'account_flag_1', 'account_flag_2', 'account_flag_3',
        'device_risk_score',
        # Signal 11: SIM Age
        'tenure_risk', 'is_new_account', 'new_acct_high_volume',
        # Composite
        'net_position', 'net_position_ratio',
        'account_type', 'segment', 'tenure_months',
        'anomaly_score_1', 'anomaly_score_2',
    ]
