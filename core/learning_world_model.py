import torch
import torch.nn as nn
import torch.nn.functional as F

class LearningWorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        """
        Models the dynamics of learning.
        Inputs:
            state: Representation of current learning state (e.g. from LearningState)
            action: Summary of parameter update (e.g. stats of delta)
            recent_loss: Recent loss value
        Outputs:
            predicted_next_loss: Scalar
            loss_trend: Scalar (slope)
            divergence_prob: Probability [0, 1]
        """
        super().__init__()
        
        # Input: state_embedding + action_embedding + recent_loss (1)
        self.input_dim = state_dim + action_dim + 1
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Heads
        self.loss_head = nn.Linear(hidden_dim, 1) # Predicted next loss
        self.trend_head = nn.Linear(hidden_dim, 1) # Predicted trend
        self.divergence_head = nn.Linear(hidden_dim, 1) # Divergence logit
        
    def forward(self, state_embedding, action_embedding, recent_loss):
        """
        Args:
            state_embedding: Tensor [B, state_dim]
            action_embedding: Tensor [B, action_dim]
            recent_loss: Tensor [B, 1]
        """
        x = torch.cat([state_embedding, action_embedding, recent_loss], dim=-1)
        feat = self.net(x)
        
        pred_loss = self.loss_head(feat)
        pred_trend = self.trend_head(feat)
        divergence_logit = self.divergence_head(feat)
        divergence_prob = torch.sigmoid(divergence_logit)
        
        return pred_loss, pred_trend, divergence_prob

    def predict_outcome(self, state_embedding, action_embedding, recent_loss):
        """External API for querying the model."""
        return self.forward(state_embedding, action_embedding, recent_loss)
