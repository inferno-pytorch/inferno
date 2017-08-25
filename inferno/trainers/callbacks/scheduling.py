from ...utils.train_utils import Duration, MovingAverage
from ...utils import python_utils as pyu
from ...utils.exceptions import assert_
from .base import Callback


class AutoLRDecay(Callback):
    def __init__(self, by_factor, patience, monitor='auto', monitor_momentum=0,
                 monitor_while='auto', exclude_param_groups=None):
        super(AutoLRDecay, self).__init__()
        # Privates
        self._patience = None
        self._last_decayed_at = {'iteration_count': None, 'epoch_count': None}
        self._last_improved_at = {'iteration_count': None, 'epoch_count': None}
        self._monitor = monitor
        self._monitor_while = monitor_while
        self._monitor_value_moving_average = MovingAverage(momentum=monitor_momentum)
        self._best_monitor_value = None
        # Publics
        self.patience = patience
        self.factor = by_factor
        self.exclude_param_groups = pyu.to_iterable(exclude_param_groups) \
            if exclude_param_groups is not None else None

    @property
    def patience(self):
        return self._patience

    @patience.setter
    def patience(self, value):
        self._patience = Duration.build_from(value)

    @property
    def monitor(self):
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value

    @property
    def monitor_value(self):
        return self.get_monitor_value()[0]

    @property
    def monitor_while(self):
        monitor_value, monitor = self.get_monitor_value()
        if self._monitor_while != 'auto':
            return self._monitor_while
        elif monitor.startswith('training_'):
            return 'training'
        elif monitor.startswith('validation_'):
            return 'validation'
        else:
            raise RuntimeError("Could not parse `monitor_while`. Please provide one manually.")

    @monitor_while.setter
    def monitor_while(self, value):
        value_mapping = {'auto': 'auto',
                         'training': 'training',
                         'validation': 'validation',
                         'validating': 'validation'}
        value = value_mapping.get(value)
        assert_(value is not None,
                "`monitor_while` must be one of {}, got {} instead."
                .format(value_mapping.keys(), value),
                ValueError)
        self._monitor_while = value

    def get_monitor_value(self):
        if self._monitor == 'auto':
            # Try to get validation error
            monitor_value = self.trainer.get_state('validation_error_averaged')
            if monitor_value is not None:
                return monitor_value, 'validation_error_averaged'
            monitor_value = self.trainer.get_state('validation_loss_averaged')
            if monitor_value is not None:
                return monitor_value, 'validation_loss_averaged'
            monitor_value = self.trainer.get_state('training_error')
            if monitor_value is not None:
                return monitor_value, 'training_error'
            monitor_value = self.trainer.get_state('training_loss')
            if monitor_value is not None:
                return monitor_value, 'training_loss'
            else:
                raise RuntimeError("Could not auto-fetch a monitor_value. "
                                   "Please specify a monitor manually.")
        else:
            monitor_value = self.trainer.get_state(self._monitor)
            assert_(monitor_value is not None,
                    "Could not fetch the specified monitor ('{}') from trainer's state."
                    .format(self._monitor),
                    ValueError)
            return monitor_value, self._monitor

    @property
    def duration_since_last_decay(self):
        since_last_decayed = {}
        if self._last_decayed_at.get('iteration_count') is None:
            since_last_decayed.update({'iteration_count': self.trainer.iteration_count})
        else:
            since_last_decayed.update(
                {'iteration_count': (self.trainer.iteration_count -
                                     self._last_decayed_at['iteration_count'])
                 })

        if self._last_decayed_at.get('epoch_count') is None:
            since_last_decayed.update({'epoch_count': self.trainer.epoch_count})
        else:
            since_last_decayed.update(
                {'epoch_count': (self.trainer.epoch_count -
                                 self._last_decayed_at['epoch_count'])
                 })
        return since_last_decayed

    @property
    def duration_since_last_improvment(self):
        since_last_improved = {}
        if self._last_improved_at.get('iteration_count') is None:
            since_last_improved.update({'iteration_count': self.trainer.iteration_count})
        else:
            since_last_improved.update(
                {'iteration_count': (self.trainer.iteration_count -
                                     self._last_improved_at['iteration_count'])
                 })

        if self._last_improved_at.get('epoch_count') is None:
            since_last_improved.update({'epoch_count': self.trainer.epoch_count})
        else:
            since_last_improved.update(
                {'epoch_count': (self.trainer.epoch_count -
                                 self._last_improved_at['epoch_count'])
                 })
        return since_last_improved

    @property
    def in_cooldown(self):
        return self.patience.match(**self.duration_since_last_improvment)

    def decay(self):
        # TODO
        # ...
        self._last_decayed_at.update({'iteration_count': self.trainer.iteration_count,
                                      'epoch_count': self.trainer.epoch_count})

    def maintain_monitor_moving_average(self):
        monitor_value = self.monitor_value
        self._monitor_value_moving_average.update(monitor_value)
        if self._best_monitor_value is None:
            self._best_monitor_value = monitor_value

    @property
    def monitor_value_has_improved(self):
        if self._monitor_value_moving_average.val is None:
            return True
        else:
            monitor_value_has_improved = \
                self._monitor_value_moving_average.val < self._best_monitor_value
            if monitor_value_has_improved:
                self._best_monitor_value = self._monitor_value_moving_average.val
                self._last_improved_at.update({'iteration_count': self.trainer.iteration_count,
                                               'epoch_count': self.trainer.epoch_count})
            return monitor_value_has_improved

    def end_of_training_iteration(self, **_):
        # Decay if we're not in cooldown (and monitoring while training)
        if self.monitor_while == 'training':
            if not self.monitor_value_has_improved and not self.in_cooldown:
                self.decay()
            self.maintain_monitor_moving_average()

    def end_of_validation_run(self, **_):
        if self.monitor_while == 'validation':
            if not self.monitor_value_has_improved and not self.in_cooldown:
                self.decay()
            self.maintain_monitor_moving_average()
