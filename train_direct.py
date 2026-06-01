import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import load_and_engineer, global_train_test_split, fit_scaler, apply_scaler, apply_smote
from models.xgb_model import MuleDetectionModel

def run():
    print("Loading data...")
    X, y, feature_names = load_and_engineer('DataSet.csv')
    X_train, X_test, y_train, y_test = global_train_test_split(X, y)
    
    print("Scaling and SMOTE...")
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(scaler, X_train)
    X_test_scaled  = apply_scaler(scaler, X_test)
    X_res, y_res = apply_smote(X_train_scaled, y_train)
    
    print("Training global model centrally (no Flower)...")
    model = MuleDetectionModel()
    model.fit(X_res, y_res, X_val=X_test_scaled, y_val=y_test)
    
    print("\nEvaluating global model...")
    metrics = model.evaluate(X_test_scaled, y_test, verbose=True)
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    model.save(str(results_dir / 'final_global_model'))
    print("Model saved to results/final_global_model.json")

if __name__ == '__main__':
    run()
