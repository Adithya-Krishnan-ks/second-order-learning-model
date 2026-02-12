import torch
import torch.nn as nn
import numpy as np
from .task_base import TaskBase

class RegressionTask(TaskBase):
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.phase = 0.0
        self.amplitude = 1.0
        self.reset()
        self._loss_fn = nn.MSELoss()

    def reset(self):
        # Shift distribution randomly
        self.phase = np.random.uniform(0, 2 * np.pi)
        self.amplitude = np.random.uniform(0.5, 2.0)

    def sample(self, batch_size: int = 32):
        x = torch.randn(batch_size, self._input_dim)
        # Sine wave task: y = A * sin(x_0 + phi)
        # We only use the first feature for the sine wave to match output_dim=1
        y = self.amplitude * torch.sin(x[:, 0:1] + self.phase)
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
