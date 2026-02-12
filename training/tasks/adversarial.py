import torch
import torch.nn as nn
from .task_base import TaskBase

class AdversarialTask(TaskBase):
    def __init__(self, input_dim: int = 5):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = 1
        self.mode = 'normal' # 'normal' or 'inverted'
        self._loss_fn = nn.MSELoss()

    def reset(self):
        # Randomly choose whether to flip the objective
        if torch.rand(1).item() < 0.5:
            self.mode = 'inverted'
        else:
            self.mode = 'normal'

    def sample(self, batch_size: int = 32):
        x = torch.randn(batch_size, self._input_dim)
        # Simple Identity task: y = sum(x)
        y = x.sum(dim=1, keepdim=True)
        
        if self.mode == 'inverted':
            y = -y # Misleading gradients if model blindly follows descent on previous tasks
            
        return x, y

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def loss_fn(self):
        return self._loss_fn
