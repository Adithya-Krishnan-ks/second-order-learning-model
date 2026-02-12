import torch


class LearningState:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.reset()

    def reset(self):
        self.step = 0
        self.loss_history = []
        self.hidden_state = torch.zeros(1, self.hidden_dim)

    def update(self, loss: float):
        self.loss_history.append(loss)
        self.step += 1
