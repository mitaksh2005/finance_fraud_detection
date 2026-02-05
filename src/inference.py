import torch
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
import os
import joblib

# Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '../outputs/models/')

LGBM_PATH = os.path.join(MODELS_DIR, 'lgbm_baseline.txt')
STD_TABNET_PATH = os.path.join(MODELS_DIR, 'tabnet_credit_baseline.zip')
MTL_TABNET_PATH = os.path.join(MODELS_DIR, 'unified_mtl_best.pth')
META_LEARNER_PATH = os.path.join(MODELS_DIR, 'meta_learner_logistic.pkl')

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
        
        # Load all models
        self.lgbm = lgb.Booster(model_file=LGBM_PATH)
        
        self.tabnet_std = TabNetClassifier()
        self.tabnet_std.load_model(STD_TABNET_PATH)
        
        input_dim = self.shared_dim
        private_dim = feature_count - self.shared_dim
        self.tabnet_mtl = MultiTaskTabNet(input_dim=input_dim, private_dim=private_dim)
        self.tabnet_mtl.load_state_dict(torch.load(MTL_TABNET_PATH, map_location=DEVICE))
        self.tabnet_mtl.eval()
        
        self.meta_learner = joblib.load(META_LEARNER_PATH)
        self.models_loaded = True

    def predict(self, df):
        # 1. LightGBM Input (Allows -999)
        X_tree = df.fillna(-999)

        # 2. Neural Input (Strictly Non-Negative)
        X_neural_np = df.fillna(0).values.astype(np.float32)
        X_neural_np[X_neural_np < 0] = 0 # Global Clip

        # Load Models if needed
        if not self.models_loaded:
            self.load_models(feature_count=df.shape[1])

        # Vocabulary Clip
        if hasattr(self.tabnet_std, 'cat_idxs'):
            for idx, dim in zip(self.tabnet_std.cat_idxs, self.tabnet_std.cat_dims):
                X_neural_np[:, idx] = np.clip(X_neural_np[:, idx], 0, dim - 1)

        # Predictions
        pred_lgbm = self.lgbm.predict(X_tree)
        pred_tabnet_std = self.tabnet_std.predict_proba(X_neural_np)[:, 1]

        # MTL Prediction
        X_tensor = torch.tensor(X_neural_np, dtype=torch.float32)
        x_shared = X_tensor[:, :self.shared_dim]
        x_private = X_tensor[:, self.shared_dim:]
        with torch.no_grad():
            _, pred_mtl = self.tabnet_mtl(x_shared, x_private)
        pred_mtl = pred_mtl.numpy().flatten()

        # Stacking
        X_stack = np.column_stack((pred_lgbm, pred_tabnet_std, pred_mtl))
        final_prob = self.meta_learner.predict_proba(X_stack)[:, 1]

        return final_prob