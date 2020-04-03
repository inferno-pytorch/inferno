from ...utils.train_utils import Frequency
from ...utils.exceptions import assert_, FrequencyValueError
from .base import Callback


class LogOutputGradients(Callback):
    """Logs the gradient of the network output"""

    def __init__(self, frequency):
        super(LogOutputGradients, self).__init__()
        self.log_every = frequency
        self.registered = False
        self.hook_handle = None

    @property
    def log_every(self):
        return self._log_every

    @log_every.setter
    def log_every(self, value):
        self._log_every = Frequency(value, 'iterations')
        assert_(self.log_every.is_consistent,
                "Log frequency is not consistent.",
                FrequencyValueError)

    def hook(self, module, grad_input, grad_output):

        #remove hook if trainer does not exits
        if self.trainer is None:
            self.hook_handle.remove()
            return

        if self.log_every.match(iteration_count=self.trainer.iteration_count,
                                epoch_count=self.trainer.epoch_count,
                                persistent=True, match_zero=True):
            self.trainer.update_state('output_gradient', grad_output[0].detach().float().clone().cpu())

    def add_hook(self):
        self.hook_handle = self.trainer.model.register_backward_hook(self.hook)

    def begin_of_fit(self, **kwargs):
        self._trainer.logger.observe_state("output_gradient",
                                           observe_while='training')
        self.add_hook()

    def begin_of_save(self, **_):
        # remove hook from model, because you can't pickle it.
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def end_of_save(self, **_):
        # add hook after model save
        self.add_hook()

