from ...utils import torch_utils as tu
from .base import Callback
from tqdm import tqdm


def silence(message):
    pass


class ProgBarTrain(Callback):
    def __init__(self, verbose=True):
        super(ProgBarTrain, self).__init__()

        self.pbar = None
        self.verbose = verbose

    def init_pbar(self):
        if self.trainer._max_num_epochs:
            self.pbar = tqdm(total=len(self.trainer.train_loader),
                             desc="Train epoch {}/{}".format(self.trainer.epoch_count, self.trainer._max_num_epochs),
                             leave=False)
        else:
            self.pbar = tqdm(total=len(self.trainer.train_loader),
                             desc="Train epoch {}".format(self.trainer.epoch_count),
                             leave=False)

    def begin_of_fit(self, **_):
        self.trainer.verbose = False

    def end_of_epoch(self, **_):
        if self.verbose:
            self.pbar.write("Train Loss: {:.3f}\tTrain Error: {:.3f}".format(tu.unwrap(self.trainer.get_state("training_loss"), as_numpy=True)[0],
                                                                     self.trainer.get_state("training_error")))
        self.pbar.close()
        self.pbar = None

    def begin_of_training_iteration(self, **_):
        if self.pbar is None:
            self.init_pbar()
        else:
            self.pbar.update(1)


class ProgBarValid(Callback):
    def __init__(self, verbose=True):
        super(ProgBarValid, self).__init__()
        self.pbar = None
        self.verbose = verbose

    def init_pbar(self):
        self.pbar = tqdm(total=len(self.trainer.validate_loader),
                         desc="Validation", leave=False)

    def begin_of_fit(self, **_):
        self.trainer.print = silence

    def end_of_validation_run(self, **_):
        if self.verbose:
            self.pbar.write("Validation Loss: {:.3f}\tValidation Error: {:.3f}".format(self.trainer.get_state("validation_loss_averaged"),
                                                                               self.trainer.get_state("validation_error_averaged")))
        self.pbar.close()
        self.pbar = None

    def begin_of_validation_iteration(self, **_):
        if self.pbar is None:
            self.init_pbar()
        else:
            self.pbar.update(1)
