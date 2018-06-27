from .base import Callback
from tqdm import tqdm
from datetime import datetime
from .console import Console


class TQDMPrinter(object):
    def __init__(self, progress):
        self._progress = progress

    def print(self, message):
        if self._progress.outer_bar is not None:
            self._progress.outer_bar.clear()
        tqdm.write(message)
        if self._progress.outer_bar is not None:
            self._progress.outer_bar.refresh()


class TQDMConsole(Console):
    def __init__(self):
        super(TQDMConsole, self).__init__(printer=TQDMPrinter(TQDMProgressBar()))


class TQDMProgressBar(Callback):
    def __init__(self, *args, **kwargs):
        super(TQDMProgressBar, self).__init__(*args, **kwargs)
        self.epoch_bar = None
        self.outer_bar = None
        self.is_training = False
        self.is_validation = False

    def bind_trainer(self, *args, **kwargs):
        super(TQDMProgressBar, self).bind_trainer(*args, **kwargs)
        self.trainer.console.toggle_progress(False)
        self.trainer.console.set_console(TQDMPrinter(self))

    def _init_epoch_bar_train(self):
        n_batch = len(self.trainer._loader_iters['train'])
        self.epoch_bar = tqdm(total=n_batch, position=1, dynamic_ncols=True)
        self.epoch_bar.update(self.trainer._batch_count)
        self.epoch_bar.set_description("Training epoch %d" % self.trainer._epoch_count)

    def print(self, message, **_):
        if self.outer_bar is not None:
            self.outer_bar.clear()
        tqdm.write("[+][{}] {}".format(str(datetime.now()), message))
        if self.outer_bar is not None:
            self.outer_bar.refresh()

    def begin_of_fit(self, max_num_epochs, **_):
        if isinstance(max_num_epochs, int):
            self.outer_bar = tqdm(total=max_num_epochs, position=0, dynamic_ncols=True)
        else:
            self.outer_bar = tqdm(total=1000, position=0, dynamic_ncols=True)
        self.outer_bar.set_description("Epochs")

    def end_of_fit(self, **_):
        if self.outer_bar is not None:
            self.outer_bar.close()
            self.outer_bar = None

    def begin_of_epoch(self, **_):
        if self.epoch_bar is not None:
            self.epoch_bar.close()

    def end_of_epoch(self, **_):
        if self.outer_bar is not None:
            self.outer_bar.update(1)

    def begin_of_training_iteration(self, **_):
        if not self.epoch_bar and 'train' in self.trainer._loader_iters.keys():
            self._init_epoch_bar_train()
            return

        if self.epoch_bar:
            self.epoch_bar.update(1)

    def begin_of_validation_iteration(self, **_):
        if self.epoch_bar:
            self.epoch_bar.update(1)

    def begin_of_training_run(self, **_):
        self.is_training = True

    def end_of_training_run(self, **_):
        self.is_training = False
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None

    def begin_of_validation_run(self, num_iterations, num_iterations_in_generator, last_validated_at_epoch, **_):
        self.is_validation = True
        nmax = num_iterations
        if not nmax:
            nmax = num_iterations_in_generator

        self.epoch_bar = tqdm(total=nmax, position=1, dynamic_ncols=True)
        self.epoch_bar.set_description("Validating epoch %d" % (last_validated_at_epoch-1))

    def end_of_validation_run(self, **_):
        self.is_validation = False
        if self.epoch_bar:
            self.epoch_bar.close()
            self.epoch_bar = None
