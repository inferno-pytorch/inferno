from ...utils.train_utils import Frequency, Duration, MovingAverage
from ...utils import python_utils as pyu
from ...utils.exceptions import assert_, NotSetError
from .base import Callback
from functools import reduce


class _Scheduler(Callback):
    def __init__(self, monitor='auto', monitor_momentum=0., monitor_while='auto'):
        super(_Scheduler, self).__init__()
        # Privates
        self._monitor_value_moving_average = MovingAverage(momentum=monitor_momentum)
        self._monitor_while = 'auto'
        self._monitor = 'auto'
        # Publics
        self.monitor = monitor
        self.monitor_while = monitor_while

    @property
    def monitor(self):
        assert_(self._monitor is not None, "Monitor is not set yet.", NotSetError)
        return self._monitor

    @monitor.setter
    def monitor(self, value):
        self._monitor = value

    @property
    def monitor_value(self):
        return self.get_monitor_value()[0]

    @property
    def monitor_while(self):
        if self._monitor_while == 'auto':
            monitor_value, monitor = self.get_monitor_value()
            if monitor.startswith('training_'):
                self._monitor_while = 'training'
            elif monitor.startswith('validation_'):
                self._monitor_while = 'validation'
            else:
                raise RuntimeError("Could not parse `monitor_while`. "
                                   "Please provide one manually.")
        return self._monitor_while

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
                self._monitor = 'validation_error_averaged'
                return monitor_value, self._monitor
            monitor_value = self.trainer.get_state('validation_loss_averaged')
            if monitor_value is not None:
                self._monitor = 'validation_loss_averaged'
                return monitor_value, self._monitor
            monitor_value = self.trainer.get_state('training_error')
            if monitor_value is not None:
                self._monitor = 'training_error'
                return monitor_value, self._monitor
            monitor_value = self.trainer.get_state('training_loss')
            if monitor_value is not None:
                self._monitor = 'training_loss'
                return monitor_value, self._monitor
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

    def maintain_monitor_moving_average(self):
        monitor_value = self.monitor_value
        self._monitor_value_moving_average.update(monitor_value)
        return monitor_value


class AutoLR(_Scheduler):
    """
    Callback to decay or hike the learning rate automatically when a specified monitor
    stops improving.

    The monitor should be decreasing, i.e. lower value --> better performance.
    """
    def __init__(self, factor, patience, required_minimum_relative_improvement=0,
                 consider_improvement_with_respect_to='best',
                 cooldown_duration=None, monitor='auto', monitor_momentum=0,
                 monitor_while='auto', exclude_param_groups=None, verbose=False):
        """
        Parameters
        ----------
        factor : float
            Factor to multiply the learning rate with when out of patience
            and not in cooldown. Setting `factor < 1` results in a LR decay,
            whereas setting `factor > 1` results in a LR hike.
        patience : str or tuple or inferno.utils.train_utils.Duration
            Specifies how long to wait for an improvement before a LR decay is triggered.
        required_minimum_relative_improvement : float
            Specifies by how much (as a fraction of the current value) the monitor should
            improve to consider the improvement significant. Leaving this to zero implies
            the monitor will be considered improving even if it's only so slightly better.
        consider_improvement_with_respect_to : {'best', 'previous'}
            While determining if the monitor has improved, the improvement is considered with
            respect to this value. Could be 'best' or 'previous'.
        cooldown_duration: str or tuple or inferno.utils.train_utils.Duration
            Wait for this duration to resume operation after having decayed LR.
        monitor : str
            Specifies the monitor. Monitor must be a trainer state, and decrease with
            increasing performance. Examples: 'validation_error', 'training_loss'.
            The monitor can be 'auto' in which case it's recommended that you specify
            `monitor_while`.
        monitor_momentum : float
            A momentum to smooth the monitor history with. Usually recommended to smooth out
            any fluctuations in the monitor value.
        monitor_while : {'auto', 'training', 'validating'}
            Whether to monitor while training or validating. If the monitor is specified
            (i.e. is not 'auto'), this can be left to 'auto'.
        exclude_param_groups : int or list
            Parameter groups to __not__ apply the LR decay on.
        verbose : bool
            Specifies if a message be printed before decaying.
        """
        super(AutoLR, self).__init__(monitor=monitor, monitor_momentum=monitor_momentum,
                                     monitor_while=monitor_while)
        # Validate
        assert_(consider_improvement_with_respect_to in ['best', 'previous'],
                "`consider_improvement_with_respect_to` must be either 'best' or 'previous', "
                "and not {}".format(consider_improvement_with_respect_to),
                ValueError)
        # Privates
        self._patience = None
        self._cooldown = None
        self._last_decayed_at = {'iteration_count': None, 'epoch_count': None}
        self._last_improved_at = {'iteration_count': None, 'epoch_count': None}
        self._best_monitor_value = None
        # Publics
        self.patience = patience
        self.cooldown_duration = cooldown_duration
        self.factor = factor
        self.required_minimum_relative_improvement = required_minimum_relative_improvement
        self.consider_improvement_with_respect_to = consider_improvement_with_respect_to
        self.exclude_param_groups = pyu.to_iterable(exclude_param_groups) \
            if exclude_param_groups is not None else None
        self.verbose = verbose

    @property
    def patience(self):
        assert_(self._patience is not None, "Patience is not set yet.", NotSetError)
        return self._patience

    @patience.setter
    def patience(self, value):
        self._patience = Duration.build_from(value)

    @property
    def cooldown_duration(self):
        return self._cooldown

    @cooldown_duration.setter
    def cooldown_duration(self, value):
        if value is not None:
            self._cooldown = Duration.build_from(value)

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
    def out_of_patience(self):
        return self.patience.match(**self.duration_since_last_improvment)

    @property
    def in_cooldown(self):
        if self.cooldown_duration is not None:
            return not self.cooldown_duration.match(**self.duration_since_last_decay)
        else:
            return False

    def decay(self):
        exclude_param_groups = \
            [] if self.exclude_param_groups is None else list(self.exclude_param_groups)
        for param_group_num, param_group in enumerate(self.trainer.optimizer.param_groups):
            if param_group_num not in exclude_param_groups:
                param_group['lr'] *= self.factor
                self.debug_print("Decayed LR of param_group {} from {} --> {}"
                                 .format(param_group_num,
                                         param_group['lr'] / self.factor,
                                         param_group['lr']))
        self._last_decayed_at.update({'iteration_count': self.trainer.iteration_count,
                                      'epoch_count': self.trainer.epoch_count})

    def maintain_monitor_moving_average(self):
        monitor_value = super(AutoLR, self).maintain_monitor_moving_average()
        if self._best_monitor_value is None:
            self._best_monitor_value = monitor_value

    @property
    def monitor_value_has_significantly_improved(self):
        if self._monitor_value_moving_average.previous is None:
            # There's nothing to compare with
            return True
        else:
            improvement_baseline = \
                self._best_monitor_value \
                if self.consider_improvement_with_respect_to == 'best' else \
                self._monitor_value_moving_average.previous
            monitor_value_has_significantly_improved = \
                self.is_significantly_less_than(self._monitor_value_moving_average.val,
                                                improvement_baseline,
                                                self.required_minimum_relative_improvement)
            self.debug_print("Is {} significantly less than {} with min_relative_delta = {}? {}."
                             .format(self._monitor_value_moving_average.val,
                                     improvement_baseline,
                                     self.required_minimum_relative_improvement,
                                     monitor_value_has_significantly_improved))
            # monitor_value_has_significantly_improved could be False, even if the current
            # moving average is less than the best monitor value, if the improvement is not
            # significant enough
            self._best_monitor_value = min([self._best_monitor_value,
                                            self._monitor_value_moving_average.val])
            if monitor_value_has_significantly_improved:
                self._last_improved_at.update({'iteration_count': self.trainer.iteration_count,
                                               'epoch_count': self.trainer.epoch_count})
            return monitor_value_has_significantly_improved

    def end_of_training_iteration(self, **_):
        # Decay if we're not in cooldown (and monitoring while training)
        if self.monitor_while == 'training':
            self.maintain_monitor_moving_average()
            if not self.monitor_value_has_significantly_improved and \
                    self.out_of_patience and not self.in_cooldown:
                if self.verbose:
                    self.trainer.console.info("Monitor '{}' has not significantly improved, decaying LR."
                                       .format(self.monitor))
                self.decay()

    def end_of_validation_run(self, **_):
        if self.monitor_while == 'validation':
            self.maintain_monitor_moving_average()
            if not self.monitor_value_has_significantly_improved \
                    and self.out_of_patience and not self.in_cooldown:
                if self.verbose:
                    self.trainer.console.info("Monitor '{}' has not significantly improved "
                                       "({} vs. {}), decaying LR."
                                       .format(self.monitor,
                                               self._monitor_value_moving_average.val,
                                               self._best_monitor_value))
                self.decay()

    @staticmethod
    def is_significantly_less_than(x, y, min_relative_delta):
        if x > y:
            return False
        relative_delta = abs(y - x) / abs(y)
        return relative_delta > min_relative_delta


class AutoLRDecay(AutoLR):
    """
    Callback to decay the learning rate automatically when a specified monitor
    stops improving.

    The monitor should be decreasing, i.e. lower value --> better performance.
    """
    pass


class DecaySpec(object):
    """A class to specify when to decay (or hike) LR and by what factor."""
    def __init__(self, duration, factor):
        # Privates
        self._matched = False
        # Publics
        self.duration = Duration.build_from(duration)
        self.factor = factor

    def match(self, iteration_count=None, epoch_count=None, when_equal_return=True):
        match_result = self.duration.match(iteration_count=iteration_count,
                                           epoch_count=epoch_count,
                                           when_equal_return=when_equal_return)
        if match_result and not self._matched:
            # First match
            self._matched = True
            return match_result
        else:
            # Already matched once (or more often)
            return False

    def new(self):
        return type(self)(self.duration, self.factor)

    @classmethod
    def build_from(cls, args):
        if isinstance(args, (list, tuple)):
            return cls(*args)
        elif isinstance(args, dict):
            return cls(**args)
        elif isinstance(args, cls):
            return args
        else:
            raise NotImplementedError("Can't build DecaySpec from {}.".format(type(args)))


class ManualLR(Callback):
    def __init__(self, decay_specs, exclude_param_groups=None):
        super(ManualLR, self).__init__()
        self.decay_specs = [DecaySpec.build_from(decay_spec)
                            for decay_spec in pyu.to_iterable(decay_specs)]
        self.exclude_param_groups = pyu.to_iterable(exclude_param_groups) \
            if exclude_param_groups is not None else None

    def match(self):
        # Find the decayspec that matched
        matched = [decay_spec
                   for decay_spec in self.decay_specs
                   if decay_spec.match(iteration_count=self.trainer.iteration_count,
                                       epoch_count=self.trainer.epoch_count)]
        if matched:
            # Allow for more than one matches; in which case the factors are multiplied
            global_factor = reduce(lambda x, y: x * y,
                                   [matched_decay_spec.factor for matched_decay_spec in matched])
            return True, global_factor
        else:
            return False, None

    def decay(self, factor):
        exclude_param_groups = \
            [] if self.exclude_param_groups is None else list(self.exclude_param_groups)
        for param_group_num, param_group in enumerate(self.trainer.optimizer.param_groups):
            if param_group_num not in exclude_param_groups:
                param_group['lr'] *= factor
                self.debug_print("Decayed LR of param_group {} from {} --> {}"
                                 .format(param_group_num,
                                         param_group['lr'] / factor,
                                         param_group['lr']))

    def end_of_training_iteration(self, **_):
        matched, global_factor = self.match()
        if matched:
            assert global_factor is not None
            self.decay(global_factor)


