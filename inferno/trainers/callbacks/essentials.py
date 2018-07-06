import numpy as np
import os
import h5py as h5
from ...utils import torch_utils as tu
from ...utils.train_utils import Frequency
from ...utils.exceptions import assert_, FrequencyValueError, NotUnwrappableError
from ...utils import python_utils as pyu
from .base import Callback


class NaNDetector(Callback):
    def end_of_training_iteration(self, **_):
        training_loss = self.trainer.get_state('training_loss')
        # Extract scalar
        if tu.is_tensor(training_loss):
            training_loss = training_loss.float()[0]
        if not np.isfinite(training_loss):
            raise RuntimeError("Loss is not finite (loss={})!".format(training_loss))


class PersistentSave(Callback):
    def __init__(self, template='checkpoint.pytorch.epoch{epoch_count}.iteration{iteration_count}'):
        super(PersistentSave, self).__init__()
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


class DumpHDF5Every(Callback):
    """Dumps intermediate training states to a HDF5 file."""
    def __init__(self, frequency, to_directory,
                 filename_template='dump.{mode}.epoch{epoch_count}.iteration{iteration_count}.h5',
                 force_dump=False, dump_after_every_validation_run=False):
        super(DumpHDF5Every, self).__init__()
        # Privates
        self._dump_every = None
        self._trainer_states_to_be_dumped_while_training = {'training_inputs',
                                                            'training_target',
                                                            'training_prediction'}
        self._trainer_states_to_be_dumped_while_validating = {'validation_inputs',
                                                              'validation_target',
                                                              'validation_prediction'}
        self._dump_cache = {}
        # Publics
        self.dump_every = frequency
        self.dump_directory = to_directory
        self.dump_filename_template = filename_template
        self.force_dump = force_dump    # hihi
        self.dump_after_every_validation_run = dump_after_every_validation_run

    @property
    def dump_every(self):
        return self._dump_every

    @dump_every.setter
    def dump_every(self, value):
        self._dump_every = Frequency.build_from(value)
        assert_(self._dump_every.is_consistent,
                "Dump frequency is not consistent.",
                FrequencyValueError)

    @property
    def dump_now(self):
        return self.dump_every.match(iteration_count=self.trainer.iteration_count,
                                     epoch_count=self.trainer.epoch_count,
                                     persistent=True, match_zero=True)

    def add_to_dump_cache(self, key, value):
        if pyu.is_listlike(value):
            for value_num, _value in enumerate(value):
                self.add_to_dump_cache("{}_{}".format(key, value_num), _value)
        else:
            self._dump_cache.update({key: value})

    def clear_dump_cache(self):
        self._dump_cache.clear()

    def dump_state(self, key, dump_while='training'):
        # Validate arguments
        keyword_mapping = {'train': 'training',
                           'training': 'training',
                           'validation': 'validating',
                           'validating': 'validating'}
        dump_while = keyword_mapping.get(dump_while)
        assert_(dump_while is not None,
                "The keyword dump_while must be one of: {}."
                .format(set(keyword_mapping.keys())),
                ValueError)
        assert_(isinstance(key, str),
                "State key must be a string, got {} instead.".format(type(key).__name__),
                TypeError)
        # Add to set of observed states
        if dump_while == 'training':
            self._trainer_states_to_be_dumped_while_training.add(key)
        elif dump_while == 'validating':
            self._trainer_states_to_be_dumped_while_validating.add(key)
        else:
            raise NotImplementedError
        return self

    def dump_states(self, keys, dump_while='training'):
        for key in keys:
            self.dump_state(key, dump_while=dump_while)
        return self

    def get_file_path(self, mode):
        # Make sure the dump directory exists
        if not os.path.exists(self.dump_directory):
            os.mkdir(self.dump_directory)
        else:
            assert_(os.path.isdir(self.dump_directory),
                    "Dump directory {} is a file.".format(self.dump_directory),
                    FileExistsError)
        filename = self.dump_filename_template.format(epoch_count=self.trainer.epoch_count,
                                                      iteration_count=self.trainer.iteration_count,
                                                      mode=mode)
        return os.path.join(self.dump_directory, filename)

    def dump(self, mode):
        with h5.File(name=self.get_file_path(mode), mode='w') as h5_file:
            for key, to_dump in self._dump_cache.items():
                if to_dump is None:
                    continue
                try:
                    to_dump = tu.unwrap(to_dump, as_numpy=True)
                except NotUnwrappableError:
                    # Can't unwrap to_dump, but let's not throw a tantrum if we're not required to
                    if not self.force_dump:
                        continue
                    else:
                        raise
                # Do the dumpin'
                h5_file.create_dataset(name=key, data=to_dump)

    def end_of_training_iteration(self, **_):
        dump_now = self.dump_now
        if dump_now:
            # To be double sure
            self.clear_dump_cache()
            # Get object to dump
            for state_name in self._trainer_states_to_be_dumped_while_training:
                self.add_to_dump_cache(state_name, self.trainer.get_state(state_name))
            # Dump
            self.dump(mode='training')
            # Clear cache
            self.clear_dump_cache()

    def end_of_validation_run(self, **_):
        if self.dump_after_every_validation_run:
            # To be double sure
            self.clear_dump_cache()
            # Get object to dump
            for state_name in self._trainer_states_to_be_dumped_while_validating:
                self.add_to_dump_cache(state_name, self.trainer.get_state(state_name))
            # Dump
            self.dump(mode='validation')
            # Clear cache
            self.clear_dump_cache()


class SaveAtBestValidationScore(Callback):
    """
    Triggers a save at the best EMA (exponential moving average) validation score.
    The basic `Trainer` has built in support for saving at the best validation score, but this
    callback might eventually replace that functionality.
    """
    def __init__(self, smoothness=0, verbose=False):
        super(SaveAtBestValidationScore, self).__init__()
        # Privates
        self._ema_validation_score = None
        self._best_ema_validation_score = None
        # Publics
        self.smoothness = smoothness
        self.verbose = verbose

    def end_of_validation_run(self, **_):
        # Get score (i.e. validation error if available, else validation loss)
        current_validation_score = self.trainer.get_state('validation_error_averaged')
        current_validation_score = self.trainer.get_state('validation_loss_averaged') \
            if current_validation_score is None else current_validation_score
        # Maintain ema
        if self._ema_validation_score is None:
            self._ema_validation_score = current_validation_score
            self._best_ema_validation_score = current_validation_score
        else:
            self._ema_validation_score = self.smoothness * self._ema_validation_score + \
                                         (1 - self.smoothness) * current_validation_score
        # This overrides the default behaviour, but reduces to it if smoothness = 0
        self.trainer._is_iteration_with_best_validation_score = \
            self._ema_validation_score < self._best_ema_validation_score
        # Trigger a save
        if self.trainer._is_iteration_with_best_validation_score:
            if self.verbose:
                self.trainer.console.info("Current smoothed validation score {} is better "
                                   "than the best smoothed validation score {}."
                                   .format(self._ema_validation_score,
                                           self._best_ema_validation_score))
            self._best_ema_validation_score = self._ema_validation_score
            self.trainer.save_now = True
        else:
            if self.verbose:
                self.trainer.console.info("Current smoothed validation score {} is not better "
                                   "than the best smoothed validation score {}."
                                   .format(self._ema_validation_score,
                                           self._best_ema_validation_score))
        # Done


class ParameterEMA(Callback):
    """Maintain a moving average of network parameters."""
    def __init__(self, momentum):
        """
        Parameters
        ----------
        momentum : float
            Momentum for the moving average. The following holds:
            `new_moving_average = momentum * old_moving_average + (1 - momentum) * value`
        """
        super(ParameterEMA, self).__init__()
        # Privates
        self._parameters = None
        # Publics
        self.momentum = momentum

    def maintain(self):
        if self._parameters is None:
            self._parameters = [p.data.new().zero_() for p in self.trainer.model.parameters()]
        for p_model, p_ema in zip(self.trainer.model.parameters(), self._parameters):
            p_ema.mul_(self.momentum).add_(p_model.data.mul(1. - self.momentum))

    def apply(self):
        assert_(self._parameters is not None,
                "Can't apply parameter EMA's: not available.",
                ValueError)
        for p_model, p_ema in zip(self.trainer.model.parameters(), self._parameters):
            p_model.data.copy_(p_ema)

    def end_of_training_iteration(self, **_):
        self.maintain()
