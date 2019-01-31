from ...utils.train_utils import Frequency
from ...utils.exceptions import assert_, FrequencyValueError
from .base import Callback
import time


class LogTrainingTime(Callback):
    """ 
        This Callback measures the elapsed time between all callback points.
        It is meant to help to analyze the training speed.
    """

    def __init__(self, frequency=1):
        super(LogTrainingTime, self).__init__()
        self.log_every = frequency
        self.times = []
        self.names = []
        self._log_now = False

    @property
    def log_every(self):
        return self._log_every

    @log_every.setter
    def log_every(self, value):
        self._log_every = Frequency(value, 'iterations')
        assert_(self.log_every.is_consistent,
                "Log frequency is not consistent.",
                FrequencyValueError)

    def add_hooks(self):
        """
            In addition to all callback points we can also add hooks to the
            model directly in order to determine the forward and backward
            times independently
        """

        def fw_hook(module, *_):
            if self.log_now():
                start_time = self.start
                self.names.append("forward pass")
                self.times.append(time.time() - start_time)

        self.hook_handle_fw = self.trainer.model.register_forward_hook(fw_hook)

        def bw_hook(module, *_):
            if self.log_now():
                start_time = self.start
                self.names.append("backward pass")
                self.times.append(time.time() - start_time)

        self.hook_handle_bw = self.trainer.model.register_backward_hook(bw_hook)

    def begin_of_fit(self, **kwargs):
        self.add_hooks()

    def begin_of_save(self, **_):
        # remove hook from model, because you can't pickle it.
        if self.hook_handle_fw is not None:
            self.hook_handle_fw.remove()
            self.hook_handle_fw = None

        if self.hook_handle_bw is not None:
            self.hook_handle_bw.remove()
            self.hook_handle_bw = None

    def log_now(self, update=False):
        if update:
            self._log_now = self.log_every.match(
                iteration_count=self.trainer.iteration_count,
                epoch_count=self.trainer.epoch_count,
                persistent=True, match_zero=True)

        return self._log_now

    def begin_of_training_iteration(self, *_, **__):
        if self.log_now(update=True):
            self.start = time.time()
            self.times = [0]
            self.names = ["begin_of_training_iteration"]

    def after_model_and_loss_is_applied(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("after_model_and_loss_is_applied")
            self.times.append(time.time() - start_time)

    def begin_of_validation_run(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("begin_of_validation_run")
            self.times.append(time.time() - start_time)

    def end_of_validation_run(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("end_of_validation_run")
            self.times.append(time.time() - start_time)

    def begin_of_validation_iteration(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("begin_of_validation_iteration")
            self.times.append(time.time() - start_time)

    def end_of_validation_iteration(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("end_of_validation_iteration")
            self.times.append(time.time() - start_time)

    def begin_of_save(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("begin_of_save")
            self.times.append(time.time() - start_time)

    def end_of_save(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("end_of_save")
            self.times.append(time.time() - start_time)

    def end_of_training_iteration(self, *_, **__):
        if self.log_now():
            start_time = self.start
            self.names.append("end_of_training_iteration")
            self.times.append(time.time() - start_time)

            print("Performance Report:")
            for i, name in enumerate(self.names[:-1]):
                print(self.times[i + 1] - self.times[i], "\t\t", name)
