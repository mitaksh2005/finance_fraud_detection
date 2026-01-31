import torch
import torch.optim as optim
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, criterion, num_epochs=20, lr=2e-2, device='cpu'):
    """
    MTL Training Loop updated for Asymmetric Feature Isolation.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Scheduler helps settle the loss after the initial fast learning
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        
        # Progress bar for your CPU training
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
        
        for i, (x_shared, x_private, y_credit, y_fraud) in pbar:
            # Move all components to device (CPU in your case)
            x_shared = x_shared.to(device)
            x_private = x_private.to(device)
            y_credit = y_credit.to(device)
            y_fraud = y_fraud.to(device)
            
            # Forward Pass with Dual Inputs
            optimizer.zero_grad()
            credit_pred, fraud_pred = model(x_shared, x_private)
            
            # Multi-Task Loss calculation
            total_loss, l_credit, l_fraud = criterion(credit_pred, y_credit, fraud_pred, y_fraud)
            
            # Backward Pass
            total_loss.backward()
            optimizer.step()
            
            train_running_loss += total_loss.item()
            pbar.set_postfix({'Loss': f"{total_loss.item():.4f}"})

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for x_shared, x_private, y_credit, y_fraud in val_loader:
                x_shared = x_shared.to(device)
                x_private = x_private.to(device)
                y_credit = y_credit.to(device)
                y_fraud = y_fraud.to(device)
                
                # Pass dual inputs to validation as well
                c_pred, f_pred = model(x_shared, x_private)
                v_loss, _, _ = criterion(c_pred, y_credit, f_pred, y_fraud)
                val_running_loss += v_loss.item()

        # Logging & Checkpointing
        epoch_loss = val_running_loss / len(val_loader)
        avg_train_loss = train_running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {epoch_loss:.4f}")
        
        # Save only if it's the best model so far
        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            # Create directory if it doesn't exist
            os.makedirs('../outputs/models/', exist_ok=True)
            torch.save(model.state_dict(), '../outputs/models/unified_mtl_best.pth')
            print("--> Model Checkpoint Saved!")

        scheduler.step()

    return model