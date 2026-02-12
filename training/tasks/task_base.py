import torch

class TaskBase:
    def __init__(self):
        pass

    def sample(self, batch_size: int = 32):
        """Returns batch_x, batch_y"""
        raise NotImplementedError

    def reset(self):
        """Resets the task internal state (e.g. shifts distribution)"""
        pass

    @property
    def input_dim(self):
        raise NotImplementedError

    @property
    def output_dim(self):
        raise NotImplementedError

    @property
    def loss_fn(self):
        raise NotImplementedError
