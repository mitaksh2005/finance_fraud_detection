import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gc

class UnifiedRiskDataset(Dataset):
    """
    MTL Dataset: Splits input into Shared and Private feature sets.
    """
    def __init__(self, X, y_credit, y_fraud, shared_dim=50):
        # Ensure data is float32 for CPU efficiency
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y_credit = torch.tensor(y_credit, dtype=torch.float32).view(-1, 1)
        self.y_fraud = torch.tensor(y_fraud, dtype=torch.float32).view(-1, 1)
        self.shared_dim = shared_dim

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Lane 1: Shared features (Financial DNA)
        x_shared = self.X[idx][:self.shared_dim] 
        
        # Lane 2: Private features (Fraud Behavioral Patterns)
        x_private = self.X[idx][self.shared_dim:] 
        
        return x_shared, x_private, self.y_credit[idx], self.y_fraud[idx]

def get_loaders(train_path, test_size=0.2, batch_size=1024, shared_dim=50):
    """
    Loads data and returns loaders designed for Asymmetric MTL.
    """
    # 1. Load Data
    df = pd.read_parquet(train_path)
    
    # 2. Define Targets
    y_fraud = df['isFraud'].values
    # Credit Proxy: Categorizing based on card type 
    y_credit = (df['card6'] == 1).astype(int).values 
    
    # 3. Define Features
    features = [col for col in df.columns if col not in ['isFraud', 'TransactionID', 'TransactionDT']]
    X = df[features]
    
    # 4. Fill NaNs 
    # Using -999 as a signal value for TabNet
    X = X.fillna(-999)
    
    # 5. Split Train/Val
    split_idx = int(len(df) * (1 - test_size))
    
    # Initialize Dataset with the shared_dim constraint
    train_ds = UnifiedRiskDataset(X[:split_idx], y_credit[:split_idx], y_fraud[:split_idx], shared_dim)
    val_ds = UnifiedRiskDataset(X[split_idx:], y_credit[split_idx:], y_fraud[split_idx:], shared_dim)
    
    # 6. Create Loaders
    # Using num_workers=0 to prevent memory overhead on 16GB RAM
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset Split: {shared_dim} Shared Features, {X.shape[1] - shared_dim} Private Features")
    
    # Cleanup
    del df, X
    gc.collect()
    
    return train_loader, val_loader