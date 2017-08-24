from .tensorboard import TensorboardLogger


def get_logger(name):
    if name in globals():
        return globals().get(name)
    else:
        raise NotImplementedError("Logger not found.")
