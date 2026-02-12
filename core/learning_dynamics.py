import torch
import torch.nn as nn
import torch.nn.functional as F

class LearningDynamicsEngine(nn.Module):
    def __init__(self, grad_dim: int, hidden_dim: int, max_update_norm: float = 1.0):
        super().__init__()
        self.gru = nn.GRUCell(grad_dim, hidden_dim)
        self.update_head = nn.Linear(hidden_dim, grad_dim)
        self.max_update_norm = max_update_norm

    def forward(self, grad_stats, state):
        # Ensure hidden_state is on the same device as the model and input
        if state.hidden_state.device != grad_stats.device:
            state.hidden_state = state.hidden_state.to(grad_stats.device)
            
        h = self.gru(grad_stats, state.hidden_state)
        delta = self.update_head(h)

        # 1. Update Magnitude Normalization (Prompt 4)
        update_norm = delta.norm()
        divergence_flag = False
        
        if update_norm > self.max_update_norm:
            delta = delta * (self.max_update_norm / (update_norm + 1e-8))
            divergence_flag = True # Flag that we clipped (potential divergence)
            
        # Optional: Trust Region logic could go here (comparing to previous params), 
        # but stateless check is just magnitude clipping.

        state.hidden_state = h # No detach here if we want to backprop through it over unrolled steps? 
        # Wait, usually for LSTMs/GRUs in meta-learning we DO want to backprop through time (BPTT).
        # But if we detach, we break the meta-gradient.
        # The prompt says: "Backpropagate meta-loss through LearningDynamicsEngine"
        # So we MUST NOT detach h if we are in the meta-training loop.
        # However, for simple rollout without meta-training, it doesn't matter.
        # Let's remove .detach() to support meta-learning.
        
        return delta, divergence_flag