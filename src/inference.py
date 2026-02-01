import torch
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
import os
import joblib  # For loading the Meta-Learner

# Add current directory to path so we can import 'model'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import MultiTaskTabNet

# --- Configuration (Must match Notebooks 04, 05, 06) ---
LGBM_PATH = '../outputs/models/lgbm_baseline.txt'
STD_TABNET_PATH = '../outputs/models/tabnet_credit_baseline.zip'
MTL_TABNET_PATH = '../outputs/models/unified_mtl_best.pth'
META_LEARNER_PATH = '../outputs/models/meta_learner_logistic.pkl'

SHARED_DIM = 50
DEVICE = torch.device('cpu')  # Inference is usually done on CPU

class RiskInferenceEngine:
    def __init__(self):
        """
        Initializes the Full Stacked Ensemble Engine.
        """
        self.shared_dim = SHARED_DIM
        self.lgbm = None
        self.tabnet_std = None
        self.tabnet_mtl = None
        self.meta_learner = None
        self.models_loaded = False

    def load_models(self, feature_count):
        """
        Loads all 4 components: LightGBM, Std TabNet, MTL TabNet, and the Meta-Learner.
        """
        print(f"⚡ Initializing Inference Engine with {feature_count} features...")

        # 1. Load LightGBM (The Tree)
        if os.path.exists(LGBM_PATH):
            self.lgbm = lgb.Booster(model_file=LGBM_PATH)
            print("✅ LightGBM loaded.")
        else:
            raise FileNotFoundError(f"Missing LightGBM model at {LGBM_PATH}")

        # 2. Load Standard TabNet (The Neural Baseline)
        if os.path.exists(STD_TABNET_PATH):
            self.tabnet_std = TabNetClassifier()
            self.tabnet_std.load_model(STD_TABNET_PATH)
            print("✅ Standard TabNet loaded.")
        else:
            raise FileNotFoundError(f"Missing Std TabNet at {STD_TABNET_PATH}")

        # 3. Load Asymmetric MTL TabNet (The Custom Arch)
        # We calculate dimensions dynamically based on input features
        input_dim = self.shared_dim
        private_dim = feature_count - self.shared_dim
        
        try:
            self.tabnet_mtl = MultiTaskTabNet(input_dim=input_dim, private_dim=private_dim)
            self.tabnet_mtl.load_state_dict(torch.load(MTL_TABNET_PATH, map_location=DEVICE))
            self.tabnet_mtl.eval()
            print("✅ Asymmetric MTL TabNet loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing MTL Model at {MTL_TABNET_PATH}")

        # 4. Load Meta-Learner (The Stacker)
        if os.path.exists(META_LEARNER_PATH):
            self.meta_learner = joblib.load(META_LEARNER_PATH)
            print("✅ Meta-Learner (Logistic Regression) loaded.")
        else:
            raise FileNotFoundError(f"Missing Meta-Learner at {META_LEARNER_PATH}")

        self.models_loaded = True

    def predict(self, df):
        """
        Runs the full Stacked Ensemble pipeline on new data.
        Returns: Numpy array of Fraud Probabilities (0.0 to 1.0)
        """
        # 1. Preprocessing (Fill NaNs)
        # Ensure we use the exact same fill value (-999) as training
        X_val = df.fillna(-999)
        
        # 2. Lazy Loading
        if not self.models_loaded:
            self.load_models(feature_count=X_val.shape[1])

        # --- STEP A: LightGBM Prediction ---
        pred_lgbm = self.lgbm.predict(X_val)

        # --- STEP B: Standard TabNet Prediction ---
        # TabNet outputs [Prob_0, Prob_1], we want column 1
        pred_tabnet_std = self.tabnet_std.predict_proba(X_val.values)[:, 1]

        # --- STEP C: Asymmetric MTL Prediction ---
        # Convert to Tensor and split for the two-lane architecture
        X_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        x_shared = X_tensor[:, :self.shared_dim]
        x_private = X_tensor[:, self.shared_dim:]
        
        with torch.no_grad():
            _, pred_mtl = self.tabnet_mtl(x_shared, x_private)
        pred_mtl = pred_mtl.numpy().flatten()

        # --- STEP D: Stacking (The Meta-Learner) ---
        # Stack predictions: [LightGBM, Std_TabNet, MTL]
        # This matches the training order in Notebook 06
        X_stack = np.column_stack((pred_lgbm, pred_tabnet_std, pred_mtl))
        
        # Final Probability from the Judge
        final_prob = self.meta_learner.predict_proba(X_stack)[:, 1]

        return final_prob

if __name__ == "__main__":
    # Test Run
    print("--- Testing Production Inference ---")
    
    try:
        # Load a tiny sample from processed data
        sample_df = pd.read_parquet('../data/processed/train.parquet').iloc[:5]
        sample_df = sample_df.drop(['isFraud', 'TransactionID', 'TransactionDT'], axis=1)
        
        engine = RiskInferenceEngine()
        risk_scores = engine.predict(sample_df)
        
        print("\nRisk Scores for 5 Sample Transactions:")
        print(risk_scores)
        print("\n✅ SYSTEM ONLINE.")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        print("Did you run Notebooks 04, 05, and 06 to generate all models?")