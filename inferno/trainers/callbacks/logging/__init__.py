
try:
    INFERNO_WITH_TENSORBOARD_LOGGER = True
    from .tensorboard import TensorboardLogger
except ImportError:
    INFERNO_WITH_TENSORBOARD_LOGGER = False


def get_logger(name):
    if name in globals():
        return globals().get(name)
    else:
        raise NotImplementedError("Logger not found.")
