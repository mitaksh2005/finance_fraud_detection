from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import uvicorn
import sys
import os

# --- PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/processed/train.parquet')

sys.path.append(BASE_DIR)
from inference import RiskInferenceEngine

app = FastAPI(title="Fraud Detection API", version="1.0")
engine = RiskInferenceEngine()

class TransactionInput(BaseModel):
    features: dict

@app.on_event("startup")
def load_model():
    print("API Startup: Loading Models...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Warning: Data file not found at {DATA_PATH}. Cannot warm up engine.")
        return

    try:
        # Load 1 row to initialize dimensions
        sample_df = pd.read_parquet(DATA_PATH).iloc[:1]
        sample_df = sample_df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
        engine.load_models(feature_count=sample_df.shape[1])
        print("✅ API Ready.")
    except Exception as e:
        print(f"Startup Failed: {e}")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is Online. Use /predict to score transactions."}

@app.post("/predict")
def predict_fraud(transaction: TransactionInput):
    try:
        input_df = pd.DataFrame([transaction.features])
        probability = engine.predict(input_df)[0]
        verdict = "BLOCK" if probability > 0.8 else ("REVIEW" if probability > 0.5 else "APPROVE")
        
        return {
            "risk_score": float(probability),
            "verdict": verdict,
            "system_status": "active"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)