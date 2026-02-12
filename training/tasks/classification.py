import torch
import torch.nn as nn
from .task_base import TaskBase

class ClassificationTask(TaskBase):
    def __init__(self, input_dim: int = 10, num_classes: int = 2, noise_level: float = 0.05):
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = num_classes # For One-hot or logits. Wait, usually cross entropy takes class index.
        # But our BaseModel outputs raw logits.
        self._num_classes = num_classes
        self.noise_level = noise_level
        
        # Fixed random weight vector defining the true boundary
        self.w = torch.randn(input_dim, num_classes)
        self._loss_fn = nn.CrossEntropyLoss()

    def reset(self):
        # Rotate the decision boundary
        self.w = torch.randn(self._input_dim, self._num_classes)

    def sample(self, batch_size: int = 32):
        x = torch.randn(batch_size, self._input_dim)
        logits = x @ self.w
        
        # Add label noise
        if self.noise_level > 0:
            noise_mask = torch.rand(batch_size) < self.noise_level
            # Randomly flip labels for noise_mask
            # We will just perturb the logits for noise to keep it simple and differentiable-ish if needed
            logits[noise_mask] += torch.randn(noise_mask.sum(), self._num_classes) * 5.0
            
        y = torch.argmax(logits, dim=1) 
        # Note: Depending on loss function, we might need one-hot. CrossEntropyLoss expects class indices.
        # But BaseModel expects output_dim. 
        # If output_dim = num_classes, model outputs logits.
        return x, y

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._num_classes

    @property
    def loss_fn(self):
        return self._loss_fn
