import torch
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
import os
import joblib

# --- DYNAMIC PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../outputs/models/')

LGBM_PATH = os.path.join(MODELS_DIR, 'lgbm_baseline.txt')
STD_TABNET_PATH = os.path.join(MODELS_DIR, 'tabnet_credit_baseline.zip')
MTL_TABNET_PATH = os.path.join(MODELS_DIR, 'unified_mtl_best.pth')
META_LEARNER_PATH = os.path.join(MODELS_DIR, 'meta_learner_logistic.pkl')

DATA_PATH = os.path.join(BASE_DIR, '../data/processed/train.parquet')

sys.path.append(BASE_DIR)
from model import MultiTaskTabNet

SHARED_DIM = 50
DEVICE = torch.device('cpu')

class RiskInferenceEngine:
    def __init__(self):
        self.shared_dim = SHARED_DIM
        self.lgbm = None
        self.tabnet_std = None
        self.tabnet_mtl = None
        self.meta_learner = None
        self.models_loaded = False

    def load_models(self, feature_count):
        print(f"⚡ Initializing Inference Engine with {feature_count} features...")

        # 1. Load LightGBM
        if os.path.exists(LGBM_PATH):
            self.lgbm = lgb.Booster(model_file=LGBM_PATH)
        else:
            raise FileNotFoundError(f"Missing LightGBM at: {LGBM_PATH}")

        # 2. Load Standard TabNet
        if os.path.exists(STD_TABNET_PATH):
            self.tabnet_std = TabNetClassifier()
            self.tabnet_std.load_model(STD_TABNET_PATH)
        else:
            raise FileNotFoundError(f"Missing Std TabNet at: {STD_TABNET_PATH}")

        # 3. Load Asymmetric MTL
        input_dim = self.shared_dim
        private_dim = feature_count - self.shared_dim
        try:
            self.tabnet_mtl = MultiTaskTabNet(input_dim=input_dim, private_dim=private_dim)
            self.tabnet_mtl.load_state_dict(torch.load(MTL_TABNET_PATH, map_location=DEVICE))
            self.tabnet_mtl.eval()
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing MTL Model at: {MTL_TABNET_PATH}")

        # 4. Load Meta-Learner
        if os.path.exists(META_LEARNER_PATH):
            self.meta_learner = joblib.load(META_LEARNER_PATH)
        else:
            raise FileNotFoundError(f"Missing Meta-Learner at: {META_LEARNER_PATH}")

        self.models_loaded = True

    def predict(self, df):
        """
        Runs the full Stacked Ensemble pipeline.
        """
        # --- 1. PREPROCESSING FOR LIGHTGBM (Tree) ---
        # Trees handle -999 natively and effectively
        X_tree = df.fillna(-999)

        # --- 2. PREPROCESSING FOR NEURAL NETS (TabNet) ---
        # Convert to numpy and ensure we are working with float32
        X_neural_np = df.fillna(0).values.astype(np.float32)

        # 3. Lazy Loading
        if not self.models_loaded:
            self.load_models(feature_count=df.shape[1])

        # --- CRITICAL FIX: SAFETY CLIP FOR EMBEDDINGS ---
        # We loop through every categorical column known to the model
        # and ensure values are >= 0 and < vocabulary size.
        if hasattr(self.tabnet_std, 'cat_idxs') and hasattr(self.tabnet_std, 'cat_dims'):
            for idx, dim in zip(self.tabnet_std.cat_idxs, self.tabnet_std.cat_dims):
                # Clip values to be safe [0, dim-1]
                # This turns -1 into 0, and any huge outliers into the max valid index
                X_neural_np[:, idx] = np.clip(X_neural_np[:, idx], 0, dim - 1)

        # --- STEP A: LightGBM Prediction ---
        pred_lgbm = self.lgbm.predict(X_tree)

        # --- STEP B: Standard TabNet Prediction ---
        pred_tabnet_std = self.tabnet_std.predict_proba(X_neural_np)[:, 1]

        # --- STEP C: Asymmetric MTL Prediction ---
        X_tensor = torch.tensor(X_neural_np, dtype=torch.float32)
        x_shared = X_tensor[:, :self.shared_dim]
        x_private = X_tensor[:, self.shared_dim:]
        
        with torch.no_grad():
            _, pred_mtl = self.tabnet_mtl(x_shared, x_private)
        pred_mtl = pred_mtl.numpy().flatten()

        # --- STEP D: Stacking ---
        X_stack = np.column_stack((pred_lgbm, pred_tabnet_std, pred_mtl))
        final_prob = self.meta_learner.predict_proba(X_stack)[:, 1]

        return final_prob

if __name__ == "__main__":
    print("--- Testing Production Inference ---")
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

        # Load sample
        sample_df = pd.read_parquet(DATA_PATH).iloc[:5]
        sample_df = sample_df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
        
        engine = RiskInferenceEngine()
        risk_scores = engine.predict(sample_df)
        
        print("\nRisk Scores:", risk_scores)
        print("\n✅ SYSTEM ONLINE.")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")