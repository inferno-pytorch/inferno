import numpy as np
from ...utils import torch_utils as tu
from .base import Callback


class NaNDetector(Callback):
    """Raises a runtime error if `training_loss` is NaN."""
    def end_of_training_iteration(self, **_):
        training_loss = self.trainer.get_state('training_loss')
        # Extract scalar
        if tu.is_tensor(training_loss):
            training_loss = training_loss.float()[0]
        if np.isnan(training_loss):
            raise RuntimeError("NaNs detected!")


class LearningRateScheduler(Callback):
    """Interfaces with pytorch learning rate schedulers."""
    def __init__(self, scheduler_class, *scheduler_args, **scheduler_kwargs):
        """
        Parameters
        ----------
        scheduler_class : type
            The scheduler class. Must be a subclass of `torch.optim.lr_scheduler._LRScheduler`
            and have a step method matching the signature of the superclass' step method.
        scheduler_args : tuple
            Positional arguments to the learning rate scheduler.
        scheduler_kwargs : dict
            Keyword arguments to the learning rate scheduler

        Warnings
        --------
        `torch.optim.lr_scheduler.ReduceLROnPlateau` is not supported because the signature of its
        `step` method does not match that of its superclass.
        """
        super(LearningRateScheduler, self).__init__()
        # TODO: Make sure scheduler_class is a subclass of torch.optim.lr_scheduler._LRScheduler
        # as soon as its available
        # Scheduler is built lazily to ensure that the trainer (with its optimizer)
        # is available
        self._scheduler = None
        self._scheduler_class = scheduler_class
        self._scheduler_args = scheduler_args
        self._scheduler_kwargs = scheduler_kwargs

    @property
    def scheduler(self):
        self.build_scheduler_maybe()
        return self._scheduler

    def build_scheduler_maybe(self):
        if self._scheduler is None:
            # Build scheduler
            if self.trainer is None:
                raise RuntimeError("Trainer is not defined for callback.")
            if not self.trainer.optimizer_is_defined:
                raise RuntimeError("Trainer has not defined an optimizer yet")
            # Build scheduler
            self._scheduler = self._scheduler_class(self.trainer.optimizer,
                                                    *self._scheduler_args,
                                                    **self._scheduler_kwargs)

    def begin_of_fit(self, **_):
        # First call must happen before the first epoch begins
        self.scheduler.step(epoch=self.trainer.epoch_count)

    def begin_of_epoch(self, **_):
        # Step
        self.scheduler.step(epoch=self.trainer.epoch_count)
