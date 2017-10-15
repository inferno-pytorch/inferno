"""Utilities for training."""
import numpy as np
from .exceptions import assert_, FrequencyTypeError, FrequencyValueError


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MovingAverage(object):
    """Computes the moving average of a given float."""
    def __init__(self, momentum=0):
        self.momentum = momentum
        self.val = None
        self.previous = None

    def reset(self):
        self.val = None

    def update(self, val):
        self.previous = self.val
        if self.val is None:
            self.val = val
        else:
            self.val = self.momentum * self.val + (1 - self.momentum) * val
        return self.val

    @property
    def relative_change(self):
        if None not in [self.val, self.previous]:
            relative_change = (self.previous - self.val) / self.previous
            return relative_change
        else:
            return None


class CLUI(object):
    """Command Line User Interface"""

    def __call__(self, f):
        def decorated(cls, *args, **kwargs):
            try:
                f(cls, *args, **kwargs)
            except KeyboardInterrupt:
                options_ = input("[!] Interrupted. Please select:\n"
                                 "[w] Save\n"
                                 "[d] Debug with PDB\n"
                                 "[q] Quit\n"
                                 "[c] Continue\n"
                                 "[?] >>> ")
                save_now = 'w' in options_
                quit_now = 'q' in options_
                debug_now = 'd' in options_
                continue_now = 'c' in options_ or not quit_now

                if save_now:
                    cls.save()

                if debug_now:
                    print("[*] Firing up PDB. The trainer instance might be accessible as 'cls'.")
                    import pdb
                    pdb.set_trace()

                if quit_now:
                    cls.print("Exiting.")
                    raise SystemExit

                if continue_now:
                    return

        return decorated


class Frequency(object):

    def __init__(self, value=None, units=None):
        # Private
        self._last_match_value = None
        self._value = None
        self._units = None
        # Public
        self.value = value
        self.units = units

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        # If value is not being set, make sure the frequency never matches muhahaha
        if value is None:
            value = np.inf
        self.assert_value_consistent(value)
        self._value = value

    UNIT_PRIORITY = 'iterations'
    VALID_UNIT_NAME_MAPPING = {'iterations': 'iterations',
                               'iteration': 'iterations',
                               'epochs': 'epochs',
                               'epoch': 'epochs'}

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        if value is None:
            value = self.UNIT_PRIORITY
        self.assert_units_consistent(value)
        self._units = self.VALID_UNIT_NAME_MAPPING.get(value)

    def assert_value_consistent(self, value=None):
        value = value or self.value
        # Make sure that value is an integer or inf
        assert_(isinstance(value, (int, float)),
                "Value must be an integer or np.inf, got {} instead."
                .format(type(value).__name__),
                FrequencyTypeError)
        if isinstance(value, float):
            assert_(value == np.inf,
                    "Provided value must be numpy.inf if a float, got {}.".format(value),
                    FrequencyValueError)

    def assert_units_consistent(self, units=None):
        units = units or self.units
        # Map
        units = self.VALID_UNIT_NAME_MAPPING.get(units)
        assert_(units is not None, "Unit '{}' not understood.".format(units),
                FrequencyValueError)

    @property
    def is_consistent(self):
        try:
            self.assert_value_consistent()
            self.assert_units_consistent()
            return True
        except (FrequencyValueError, FrequencyTypeError):
            return False

    def epoch(self):
        self.units = 'epochs'
        return self

    def iteration(self):
        self.units = 'iterations'
        return self

    @property
    def by_epoch(self):
        return self.units == 'epochs'

    @property
    def by_iteration(self):
        return self.units == 'iterations'

    def every(self, value):
        self.value = value
        return self

    def match(self, iteration_count=None, epoch_count=None, persistent=False, match_zero=True):
        match_value = {'iterations': iteration_count, 'epochs': epoch_count}.get(self.units)
        if not match_zero and match_value == 0:
            match = False
        else:
            match = match_value is not None and \
                    self.value != np.inf and \
                    match_value % self.value == 0
        if persistent and match and self._last_match_value == match_value:
            # Last matched value is the current matched value, i.e. we've matched once already,
            # and don't need to match again
            match = False
        if match:
            # Record current match value as the last known match value to maintain persistency
            self._last_match_value = match_value
        return match

    def __str__(self):
        return "{} {}".format(self.value, self.units)

    def __repr__(self):
        return "{}(value={}, units={})".format(type(self).__name__, self.value, self.units)

    @classmethod
    def from_string(cls, string):
        assert_(isinstance(string, str), "`string` must be a string, got {} instead."
                .format(type(string).__name__), TypeError)
        if string == 'never':
            return cls(np.inf, 'iterations')
        else:
            value_and_unit = string.split(' ')
            assert_(len(value_and_unit) == 2,
                    "Was expecting a string 'value units' with one white-space "
                    "between 'value' and 'units'.", ValueError)
            value, unit = value_and_unit
            value = np.inf if value == 'inf' else int(value)
            return cls(value, unit)

    @classmethod
    def build_from(cls, args, priority='iterations'):
        if isinstance(args, int):
            return cls(args, priority)
        elif isinstance(args, (tuple, list)):
            return cls(*args)
        elif isinstance(args, Frequency):
            return args
        elif isinstance(args, str):
            return cls.from_string(args)
        else:
            raise NotImplementedError


class Duration(Frequency):
    """Like frequency, but measures a duration."""
    def match(self, iteration_count=None, epoch_count=None, when_equal_return=False, **_):
        match_value = {'iterations': iteration_count, 'epochs': epoch_count}.get(self.units)
        assert_(match_value is not None,
                "Could not match duration because {} is not known.".format(self.units),
                ValueError)
        if match_value == self.value:
            return when_equal_return
        return match_value > self.value

    def compare(self, iteration_count=None, epoch_count=None):
        compare_value = {'iterations': iteration_count, 'epochs': epoch_count}.get(self.units)
        assert_(compare_value is not None,
                "Could not match duration because {} is not known.".format(self.units),
                ValueError)
        compared = {'iterations': None, 'epochs': None}
        compared.update({self.units: self.value - compare_value})
        return compared

    def __sub__(self, other):
        assert_(isinstance(other, Duration),
                "Object of type {} cannot be subtracted from "
                "a Duration object.".format(type(other)),
                TypeError)
        assert_(other.units == self.units,
                "The Duration objects being subtracted must have the same units.",
                ValueError)
        return Duration(value=(self.value - other.value), units=self.units)


class NoLogger(object):
    def __init__(self, logdir=None):
        self.logdir = logdir

    def log_value(self, *kwargs):
        pass


def set_state(module, key, value):
    """Writes `key`-`value` pair to `module`'s state hook."""
    if hasattr(module, '_state_hooks'):
        state_hooks = getattr(module, '_state_hooks')
        assert isinstance(state_hooks, dict), \
            "State hook (i.e. module._state_hooks) is not a dictionary."
        state_hooks.update({key: value})
    else:
        setattr(module, '_state_hooks', {key: value})
    return module


def get_state(module, key, default=None):
    """Gets key from `module`'s state hooks."""
    return getattr(module, '_state_hooks', {}).get(key, default)
