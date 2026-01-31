import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedMultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, credit_weight=0.5, fraud_weight=0.5):
        """
        Combines Credit Loss and Fraud Focal Loss.
        alpha & gamma: Hyperparameters for Focal Loss to handle 3.5% fraud imbalance.
        """
        super(WeightedMultiTaskLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.credit_weight = credit_weight
        self.fraud_weight = fraud_weight

    def focal_loss(self, inputs, targets):
        """
        Focal Loss specifically for the highly imbalanced Fraud task.
        """
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # Prevents easy examples from dominating
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

    def forward(self, credit_pred, credit_target, fraud_pred, fraud_target):
        # 1. Standard Loss for Credit Risk
        loss_credit = F.binary_cross_entropy(credit_pred, credit_target)
        
        # 2. Focal Loss for Fraud Detection
        loss_fraud = self.focal_loss(fraud_pred, fraud_target)
        
        # 3. Weighted Total (Balance the two heads)
        total_loss = (self.credit_weight * loss_credit) + (self.fraud_weight * loss_fraud)
        
        return total_loss, loss_credit, loss_fraud