import torch
from .base import Logger
try:
    import tensorboard_logger as tflogger
except ImportError:
    tflogger = None


class BasicTensorboardLogger(Logger):
    def __init__(self, log_directory=None):
        # Make sure tensorboard_logger is availble
        assert tflogger is not None
        super(BasicTensorboardLogger, self).__init__(log_directory=log_directory)
        self._logger = None

    @property
    def logger(self):
        if self._logger is None:
            self._logger = tflogger.Logger(logdir=self.log_directory)
        return self._logger

    def end_of_training_iteration(self, **_):
        # Fetch from trainer
        training_loss = self.trainer.get_state('training_loss')
        training_error = self.trainer.get_state('training_error')
        # Extract floats from torch tensors if necessary
        if torch.is_tensor(training_loss):
            training_loss = training_loss[0]
        if torch.is_tensor(training_error):
            training_error = training_error[0]
        # Log
        self.logger.log_value('training_loss', training_loss, self.trainer.iteration_count)
        self.logger.log_value('training_error', training_error, self.trainer.iteration_count)

    def end_of_validation_run(self, **meters):
        validation_loss_meter = meters.get('validation_loss_meter')
        validation_error_meter = meters.get('validation_error_meter')
        if validation_loss_meter is not None:
            self.logger.log_value('validation_loss',
                                  validation_loss_meter.avg,
                                  self.trainer.iteration_count)
        if validation_error_meter is not None:
            self.logger.log_value('validation_error_meter',
                                  validation_error_meter.avg,
                                  self.trainer.iteration_count)

