import numpy as np
import os
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


class SaveFilenameCallback(Callback):
    def __init__(self, template='checkpoint.pytorch.epoch{epoch_count}.iteration{iteration_count}'):
        self.template = template

    def begin_of_save(self, **kwargs):
        self._orig_checkpoint_filename = self.trainer._checkpoint_filename
        self.trainer._checkpoint_filename = self.template.format(**kwargs)

    def end_of_save(self, save_to_directory, **_):
        orig_checkpoint_path = os.path.join(save_to_directory, self._orig_checkpoint_filename)

        if os.path.lexists(orig_checkpoint_path):
            os.remove(orig_checkpoint_path)
        os.symlink(self.trainer._checkpoint_filename, orig_checkpoint_path)

        self.trainer._checkpoint_filename = self._orig_checkpoint_filename
