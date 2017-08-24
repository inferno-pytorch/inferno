"""Utility functions with no external dependencies."""
import signal


def is_listlike(x):
    return isinstance(x, (list, tuple))


def to_iterable(x):
    return [x] if not is_listlike(x) else x


def from_iterable(x):
    return x[0] if (is_listlike(x) and len(x) == 1) else x


def robust_len(x):
    return len(x) if is_listlike(x) else 1


def as_tuple_of_len(x, len_):
    if is_listlike(x):
        assert len(x) == len_, \
            "Listlike object of len {} can't be returned " \
            "as a tuple of length {}.".format(len(x), len_)
        return tuple(x)
    else:
        return (x,) * len_


def has_callable_attr(object_, name):
    return hasattr(object_, name) and callable(getattr(object_, name))


def is_maybe_list_of(check_function):
    def decorated_function(object_, **kwargs):
        if isinstance(object_, (list, tuple)):
            return all([check_function(_object, **kwargs) for _object in object_])
        else:
            return check_function(object_, **kwargs)
    return decorated_function


class delayed_keyboard_interrupt(object):
    """
    Delays SIGINT over critical code.
    Borrowed from:
    https://stackoverflow.com/questions/842557/
    how-to-prevent-a-block-of-code-from-being-interrupted-by-keyboardinterrupt-in-py
    """
    # PEP8: Context manager class in lowercase
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def get_config_for_name(config, name):
    config_for_name = {}
    for key, val in config.items():
        if isinstance(val, dict) and name in val:
            # we leave the slicing_config validation to classes higher up in MRO
            config_for_name.update({key: val.get(name)})
        else:
            config_for_name.update({key: val})
    return config_for_name
