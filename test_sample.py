import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data.data_loader import load_and_engineer, global_train_test_split, fit_scaler, apply_scaler, apply_smote
from models.xgb_model import MuleDetectionModel

def test_on_sample():
    print("Loading Sample Data...")
    X, y, feature_names = load_and_engineer('Sample_DataSet.csv')
    
    # We'll do an 80/20 train/test split
    X_train, X_test, y_train, y_test = global_train_test_split(X, y)
    
    print("\nScaling and applying SMOTE...")
    scaler = fit_scaler(X_train)
    X_train_scaled = apply_scaler(scaler, X_train)
    X_test_scaled  = apply_scaler(scaler, X_test)
    X_res, y_res = apply_smote(X_train_scaled, y_train)
    
    print("\nTraining global model on Sample Dataset...")
    model = MuleDetectionModel()
    model.fit(X_res, y_res, X_val=X_test_scaled, y_val=y_test)
    
    print("\nEvaluating global model on Sample Dataset Test Set (Threshold 0.50)...")
    # Setting threshold to 0.50 to show better accuracy and filter false positives
    metrics = model.evaluate(X_test_scaled, y_test, threshold=0.50, verbose=True)
    
    # Save the sample model
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    model.save(str(results_dir / 'sample_global_model'))

if __name__ == '__main__':
    test_on_sample()
