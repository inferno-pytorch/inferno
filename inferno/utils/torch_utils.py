import numpy as np
import torch
from torch.autograd import Variable

from .python_utils import delayed_keyboard_interrupt


def unwrap(tensor_or_variable, as_numpy=False):
    if isinstance(tensor_or_variable, (list, tuple)):
        return type(tensor_or_variable)([unwrap(_t) for _t in tensor_or_variable])
    elif isinstance(tensor_or_variable, Variable):
        tensor = tensor_or_variable.data
    elif torch.is_tensor(tensor_or_variable):
        tensor = tensor_or_variable
    elif isinstance(tensor_or_variable, np.ndarray):
        return tensor_or_variable
    elif isinstance(tensor_or_variable, (float, int)):
        return tensor_or_variable
    else:
        raise NotImplementedError
    # Transfer to CPU, and then to numpy, and return
    with delayed_keyboard_interrupt():
        if as_numpy:
            return tensor.cpu().numpy()
        else:
            return tensor.cpu()


def is_tensor(object_):
    missed_tensor_classes = {torch.HalfTensor}
    return torch.is_tensor(object_) or type(object_) in missed_tensor_classes


def is_image_tensor(object_):
    return is_tensor(object_) and object_.dim() == 4


def is_volume_tensor(object_):
    return is_tensor(object_) and object_.dim() == 5