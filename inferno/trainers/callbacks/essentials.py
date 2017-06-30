import numpy as np
from ...utils import torch_utils as tu
from .base import Callback


class NaNDetector(Callback):
    def end_of_training_iteration(self, **_):
        training_loss = self.trainer.get_state('training_loss')
        # Extract scalar
        if tu.is_tensor(training_loss):
            training_loss = training_loss.float()[0]
        if np.isnan(training_loss):
            raise RuntimeError("NaNs detected!")
