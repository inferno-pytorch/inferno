"""Utility functions with no external dependencies."""
import signal
import warnings
import functools
import inspect
import os


def ensure_dir(directory):
    """ensure the existence of e directory at a given path

        If the directory does not exist it is created

    Args:
        directory (str): path of the directory

    Returns:
        str: path of the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def require_dict_kwargs(kwargs, msg=None):
    """ Ensure arguments passed kwargs are either None or a dict.
        If arguments are neither a dict nor None a RuntimeError
        is thrown
    Args:
        kwargs (object): possible dict or None
        msg (None, optional): Error msg

    Returns:
        dict: kwargs dict

    Raises:
        RuntimeError: if the passed value is neither a dict nor None
            this error is raised
    """
    if kwargs is None:
        return dict()
    elif isinstance(kwargs, dict):
        return kwargs
    else:
        if msg is None:
            raise RuntimeError("value passed as keyword argument dict is neither None nor a dict")
        else:
            raise RuntimeError("%s"%str(msg))


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

string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Borrowed from
    https://stackoverflow.com/questions/2536307/
    decorators-in-the-python-standard-lib-deprecated-specifically
    by Laurent LAPORTE
    https://stackoverflow.com/users/1513933/laurent-laporte

    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
