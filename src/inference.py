import torch
import lightgbm as lgb
import pandas as pd
import numpy as np
import sys
import os
from model import MultiTaskTabNet

# Configuration
TABNET_PATH = '../outputs/models/unified_mtl_best.pth'
LGBM_PATH = '../outputs/models/lgbm_model.txt'
SHARED_DIM = 50
DEVICE = torch.device('cpu') # Inference is usually done on CPU

class RiskInferenceEngine:
    def __init__(self, input_dim=None, private_dim=None):
        """
        Initializes the Hybrid Engine by loading both models.
        """
        self.shared_dim = SHARED_DIM
        
        # We need dimensions to initialize TabNet. 
        # If not provided, we assume standard dimensions from training (446 total)
        # 446 total - 50 shared = 396 private (Example)
        if input_dim is None: 
            # Fallback or strict requirement. 
            # Better to require it or infer from a dummy load if needed.
            # For now, we will require the user to pass the feature count or instantiate lazily.
            pass

        self.tabnet = None
        self.lgbm = None
        self.models_loaded = False

    def load_models(self, feature_count):
        """
        Loads models dynamically based on the input feature count.
        """
        input_dim = self.shared_dim
        private_dim = feature_count - self.shared_dim
        
        print(f"Initializing Engine with {feature_count} features ({input_dim} Shared + {private_dim} Private)...")

        # 1. Load TabNet
        try:
            self.tabnet = MultiTaskTabNet(input_dim=input_dim, private_dim=private_dim)
            self.tabnet.load_state_dict(torch.load(TABNET_PATH, map_location=DEVICE))
            self.tabnet.eval()
            print("✅ TabNet loaded successfully.")
        except FileNotFoundError:
            print(f"❌ Error: TabNet weights not found at {TABNET_PATH}")
            sys.exit(1)
        
        # 2. Load LightGBM
        if os.path.exists(LGBM_PATH):
            self.lgbm = lgb.Booster(model_file=LGBM_PATH)
            print("✅ LightGBM loaded successfully.")
        else:
            print(f"⚠️ Warning: LightGBM model not found at {LGBM_PATH}. Running in TabNet-only mode.")
            self.lgbm = None
            
        self.models_loaded = True

    def predict(self, df):
        """
        Accepts a DataFrame, ensures models are loaded, and returns Risk Probability.
        """
        # 1. Preprocessing
        X_val = df.fillna(-999)
        
        # 2. Lazy Loading (Initialize on first predict call if not ready)
        if not self.models_loaded:
            self.load_models(feature_count=X_val.shape[1])

        # 3. TabNet Prediction
        X_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        x_shared = X_tensor[:, :self.shared_dim]
        x_private = X_tensor[:, self.shared_dim:]
        
        with torch.no_grad():
            _, tabnet_prob = self.tabnet(x_shared, x_private)
            tabnet_prob = tabnet_prob.numpy().flatten()
            
        # 4. LightGBM Prediction & Ensemble
        if self.lgbm:
            lgbm_prob = self.lgbm.predict(X_val)
            # Weighted Hybrid Score
            final_prob = (0.4 * tabnet_prob) + (0.6 * lgbm_prob)
        else:
            final_prob = tabnet_prob
            
        return final_prob

if __name__ == "__main__":
    # Example Usage
    print("--- Testing Hybrid Inference Engine ---")
    
    # Create a dummy row to test dimensions (based on your dataset size)
    # This is just for testing the script execution
    try:
        # Load a tiny sample just to get columns/shape
        sample_df = pd.read_parquet('../data/processed/train.parquet').iloc[:5]
        sample_df = sample_df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
        
        engine = RiskInferenceEngine()
        risk_scores = engine.predict(sample_df)
        
        print("\nTest Predictions:")
        print(risk_scores)
        print("\n✅ System is ready for production.")
        
    except Exception as e:
        print(f"\n⚠️ Could not run test: {e}")
        print("Ensure data is available in ../data/processed/")