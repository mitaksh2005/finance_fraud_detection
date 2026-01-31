import torch
import pandas as pd
import numpy as np
from model import MultiTaskTabNet
import gc

class RiskInferenceEngine:
    def __init__(self, model_path, input_dim, cat_dims, cat_idxs):
        """
        Loads the trained MTL weights and prepares the Decision Engine.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultiTaskTabNet(input_dim, cat_dims, cat_idxs)
        
        # Load the saved .pth weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"Unified Risk Model loaded successfully on {self.device}")

    def predict_risk(self, feature_vector):
        """
        Takes a processed feature vector and returns Credit and Fraud scores.
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            x = torch.tensor(feature_vector, dtype=torch.float32).to(self.device)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            
            credit_prob, fraud_prob = self.model(x)
            
        return credit_prob.item(), fraud_prob.item()

    def decision_engine(self, credit_score, fraud_prob, fraud_threshold=0.8, credit_threshold=0.3):
        """
        The Logic Flow from your flowchart:
        - High Fraud -> Auto Reject
        - Low Fraud + Low Credit -> Auto Reject
        - Low Fraud + High Credit -> Auto Approve
        - Others -> Manual Review
        """
        if fraud_prob >= fraud_threshold:
            return "🔴 AUTO-REJECT: High Fraud Risk"
        
        if fraud_prob < 0.2 and credit_score > (1 - credit_threshold):
            return "🟢 AUTO-APPROVE: Low Risk Profile"
        
        if credit_score < 0.4:
            return "🔴 AUTO-REJECT: High Credit Risk"
            
        return "🟡 MANUAL REVIEW: Ambiguous Case"

# --- Example Usage Script ---
if __name__ == "__main__":
    # These params must match your training setup from 04/05.ipynb
    INPUT_DIM = 446 
    # Placeholder dims/idxs - in practice, import these from your config or saved metadata
    
    engine = RiskInferenceEngine(
        model_path='../outputs/models/unified_mtl_best.pth',
        input_dim=INPUT_DIM,
        cat_dims=[], # Provide actual cat_dims here
        cat_idxs=[]  # Provide actual cat_idxs here
    )

    # Simulate a single user transaction from your test set
    test_data = pd.read_parquet('../data/processed/test_engineered.parquet').iloc[0]
    features = test_data.drop(['TransactionID', 'TransactionDT']).values.astype(np.float32)
    
    c_score, f_prob = engine.predict_risk(features)
    decision = engine.decision_engine(c_score, f_prob)
    
    print(f"\n--- Decision Report ---")
    print(f"Credit Probability: {c_score:.4f}")
    print(f"Fraud Probability:  {f_prob:.4f}")
    print(f"Action: {decision}")