import torch
import torch.nn as nn


class MetaController(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3) # outputs: loss_weight, gate, noise_scale
        )

    def forward(self, metrics):
        out = torch.tanh(self.net(metrics))
        
        # Parse outputs (Prompt 6)
        # 1. Loss Weight: [-1, 1] -> [0, 2] (scale factor)
        loss_weight = out[..., 0] + 1.0 
        
        # 2. Gate: [-1, 1] -> [0, 1] (sigmoid-like behavior via tanh+1 / 2)
        gate = (out[..., 1] + 1.0) / 2.0
        
        # 3. Noise Scale: [-1, 1] -> [0, 0.1] (small noise)
        noise_scale = (out[..., 2] + 1.0) * 0.05
        
        return loss_weight, gate, noise_scale
