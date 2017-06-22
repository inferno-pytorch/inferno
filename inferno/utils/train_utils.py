"""Utilities for training."""


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
        self.value = value
        self.units = units

    @property
    def is_consistent(self):
        return isinstance(self.value, int) and self.units in ['epochs', 'iterations']

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
        assert isinstance(value, int), "Frequency must be an integer."
        self.value = value
        return self

    def match(self, iteration_count=None, epoch_count=None):
        match_value = {'iterations': iteration_count, 'epochs': epoch_count}.get(self.units)
        return match_value is not None and match_value % self.value == 0

    def __mod__(self, other):
        if isinstance(other, int):
            return self.value % other
        elif isinstance(other, Frequency):
            # Check if units match
            assert self.units == other.units, "Units don't match."
            return self.value % other.value
        else:
            raise NotImplementedError("Can't compute modulo of a {} with Frequency."
                                      .format(type(other)))

    def __str__(self):
        return "{} {}".format(self.value, self.units)

    def __repr__(self):
        return "Frequency(value={}, units={})".format(self.value, self.units)

    @classmethod
    def from_string(cls, string):
        value, unit = string.split(' ')
        assert unit in ['epochs', 'iterations']
        return cls(int(value), unit)

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


class NoLogger(object):
    def __init__(self, logdir=None):
        self.logdir = logdir

    def log_value(self, *kwargs):
        pass


