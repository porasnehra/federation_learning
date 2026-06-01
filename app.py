import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import numpy as np

from models.xgb_model import MuleDetectionModel
from utils.feature_engineering import engineer_features, get_feature_names
from predict import get_risk_tier, explain_signals

app = FastAPI(title="Mule Account Detection API")

# Initialize and load model globally so it's ready when requests come in
model = MuleDetectionModel()
MODEL_PATH = Path("results/final_global_model.json")

# Try to load either the final global model or the sample one we just made
if MODEL_PATH.exists():
    model.load(str(MODEL_PATH.with_suffix('')))
elif Path("results/sample_global_model.json").exists():
    model.load("results/sample_global_model")
else:
    print("Warning: Model file not found. Please train the model first.")

class PredictionRequest(BaseModel):
    # This accepts the raw JSON data sent by your Flutter app
    features: Dict[str, Any]

@app.get("/")
def read_root():
    return {"status": "Active", "message": "Mule Detection API is running. Send POST request to /predict"}

@app.post("/predict")
def predict_risk(request: PredictionRequest):
    try:
        # 1. Convert the incoming JSON dictionary from Flutter into a Pandas DataFrame (1 row)
        df = pd.DataFrame([request.features])
        
        # 2. Run your existing Feature Engineering pipeline
        features_df = engineer_features(df)
        feature_cols = get_feature_names()
        
        # Ensure all required features exist even if Flutter left some out (fill with 0)
        for col in feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
                
        X = features_df[feature_cols].values.astype(np.float32)
        
        # 3. Predict the Mule Probability
        prob = float(model.predict_proba(X)[0])
        tier = get_risk_tier(prob)
        signals = explain_signals(features_df.iloc[0])
        
        # 4. Return the result back to Flutter
        return {
            "mule_probability": round(prob, 4),
            "risk_tier": tier,
            "flagged": prob >= 0.35, # default threshold
            "signals_triggered": len(signals),
            "signals": signals
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    # Render provides the PORT environment variable. If not found, default to 10000.
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
