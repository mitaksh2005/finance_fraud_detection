import torch
import numpy as np
import pandas as pd
from data_loader import get_loaders
from model import MultiTaskTabNet
from loss_functions import WeightedMultiTaskLoss
from trainer import train_model
import gc
import os

# ==========================================
# 1. CONFIGURATION (Optimized for 16GB RAM)
# ==========================================
CONFIG = {
    'TRAIN_PATH': '../data/processed/train.parquet', 
    'MODEL_SAVE_PATH': '../outputs/models/unified_mtl_best.pth',
    'BATCH_SIZE': 1024,      # Stable for CPU; drop to 512 if RAM hits >90%
    'EPOCHS': 20,
    'LR': 1e-3,              # Balanced rate for multi-task stability
    'SHARED_DIM': 50,        # First 50 columns are Shared (DNA)
    'FRAUD_WEIGHT': 0.8,      # Increased to 0.8 to recover from the AUC drop
    'CREDIT_WEIGHT': 0.2,     # Credit is the secondary, "easier" task
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def run_pipeline():
    print(f"--- Starting Asymmetric Unified Risk Training on {CONFIG['DEVICE']} ---")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(CONFIG['MODEL_SAVE_PATH']), exist_ok=True)

    # 2. DATA LOADING
    # Now explicitly passing shared_dim to create the two-lane split
    train_loader, val_loader = get_loaders(
        CONFIG['TRAIN_PATH'],
        batch_size=CONFIG['BATCH_SIZE'],
        shared_dim=CONFIG['SHARED_DIM']
    )

    # 3. DYNAMIC METADATA
    # Extracting dimensions from the first batch (x_shared, x_private, y_c, y_f)
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[1]    # Shared features
    private_dim = sample_batch[1].shape[1]  # Private features
    
    print(f"Shared Features:  {input_dim}")
    print(f"Private Features: {private_dim}")

    # 4. INITIALIZE ASYMMETRIC MODEL
    model = MultiTaskTabNet(
        input_dim=input_dim,
        private_dim=private_dim
    ).to(CONFIG['DEVICE'])

    # 5. LOSS & OPTIMIZER
    # Using higher fraud_weight to force the model to prioritize the harder task
    criterion = WeightedMultiTaskLoss(
        alpha=0.25, 
        gamma=2.0, 
        credit_weight=CONFIG['CREDIT_WEIGHT'],
        fraud_weight=CONFIG['FRAUD_WEIGHT']
    )

    # 6. EXECUTION
    print("Beginning Training Loop...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=CONFIG['EPOCHS'],
        lr=CONFIG['LR'],
        device=CONFIG['DEVICE']
    )

    print(f"--- Pipeline Complete. Model saved to {CONFIG['MODEL_SAVE_PATH']} ---")

if __name__ == "__main__":
    gc.collect()
    run_pipeline()