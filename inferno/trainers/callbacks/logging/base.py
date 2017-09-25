import os
from ..base import Callback


class Logger(Callback):
    """
    A special callback for logging.

    Loggers are special because they're required to be serializable, whereas other
    callbacks have no such guarantees. In this regard, they jointly handled by
    trainers and the callback engine.
    """
    def __init__(self, log_directory=None):
        super(Logger, self).__init__()
        self._log_directory = None
        if log_directory is not None:
            self.set_log_directory(log_directory)

    @property
    def log_directory(self):
        if self._log_directory is not None:
            return self._log_directory
        elif self.trainer is not None and self.trainer._log_directory is not None:
            return self.trainer._log_directory
        else:
            raise RuntimeError("No log directory found.")

    @log_directory.setter
    def log_directory(self, value):
        self.set_log_directory(value)

    def set_log_directory(self, log_directory):
        assert isinstance(log_directory, str)
        if not os.path.isdir(log_directory):
            assert not os.path.exists(log_directory)
            os.makedirs(log_directory)
        self._log_directory = log_directory
        return self
