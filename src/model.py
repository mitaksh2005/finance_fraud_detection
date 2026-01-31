import torch
import torch.nn as nn
from pytorch_tabnet.tab_network import TabNetNoEmbeddings

class MultiTaskTabNet(nn.Module):
    def __init__(self, input_dim, private_dim, n_d=64, n_a=64, n_steps=5):
        """
        Asymmetric MTL Architecture: 
        - Credit Head sees ONLY shared features via Encoder.
        - Fraud Head sees Shared features + Private features.
        """
        super(MultiTaskTabNet, self).__init__()
        
        # 1. THE SHARED ENCODER (Backbone)
        # We only pass the shared features through the attention mechanism
        self.encoder = TabNetNoEmbeddings(
            input_dim=input_dim,
            output_dim=n_d,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            epsilon=1e-15,
            virtual_batch_size=128,
            momentum=0.02,
            mask_type="entmax"
        )
        
        # 2. CREDIT RISK HEAD
        # Focuses on the general financial profile (the n_d output)
        self.credit_head = nn.Sequential(
            nn.Linear(n_d, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 3. FRAUD DETECTION HEAD (The Asymmetric "Private Lane")
        # Takes the shared representation (n_d) AND the raw private features
        self.fraud_head = nn.Sequential(
            nn.Linear(n_d + private_dim, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3), # Higher dropout for the complex task
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x_shared, x_private):
        # Step 1: Pass only shared features through TabNet
        shared_rep, _ = self.encoder(x_shared)
        
        # Step 2: Credit Head relies purely on shared knowledge
        credit_prob = self.credit_head(shared_rep)
        
        # Step 3: Fraud Head concatenates shared and private info
        fraud_combined_input = torch.cat([shared_rep, x_private], dim=1)
        fraud_prob = self.fraud_head(fraud_combined_input)
        
        return credit_prob, fraud_prob

    def get_attention_masks(self, x_shared):
        return self.encoder.forward_masks(x_shared)