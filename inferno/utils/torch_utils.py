import numpy as np
import torch
from torch.autograd import Variable

from .python_utils import delayed_keyboard_interrupt


def unwrap(tensor_or_variable, to_cpu=True, as_numpy=False):
    if isinstance(tensor_or_variable, (list, tuple)):
        return type(tensor_or_variable)([unwrap(_t, to_cpu=to_cpu, as_numpy=as_numpy)
                                         for _t in tensor_or_variable])
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
    # Transfer to CPU if required
    if to_cpu:
        with delayed_keyboard_interrupt():
            tensor = tensor.cpu()
    # Convert to numpy if required
    if as_numpy:
        return tensor.cpu().numpy()
    else:
        return tensor


def is_tensor(object_):
    missed_tensor_classes = {torch.HalfTensor}
    return torch.is_tensor(object_) or type(object_) in missed_tensor_classes


def is_image_tensor(object_):
    return is_tensor(object_) and object_.dim() == 4


def is_volume_tensor(object_):
    return is_tensor(object_) and object_.dim() == 5


def is_image_or_volume_tensor(object_):
    return is_image_tensor(object_) or is_volume_tensor(object_)


def is_matrix_tensor(object_):
    return is_tensor(object_) and object_.dim() == 2


def is_scalar_tensor(object_):
    return is_tensor(object_) and object_.dim() == 1 and object_.numel() == 1


def where(condition, if_true, if_false):
    """
    Torch equivalent of numpy.where.

    Parameters
    ----------
    condition : torch.ByteTensor or torch.cuda.ByteTensor or torch.autograd.Variable
        Condition to check.
    if_true : torch.Tensor or torch.cuda.Tensor or torch.autograd.Variable
        Output value if condition is true.
    if_false: torch.Tensor or torch.cuda.Tensor or torch.autograd.Variable
        Output value if condition is false

    Returns
    -------
    torch.Tensor

    Raises
    ------
    AssertionError
        if if_true and if_false are not both variables or both tensors.
    AssertionError
        if if_true and if_false don't have the same datatype.
    """
    if isinstance(if_true, Variable) or isinstance(if_false, Variable):
        assert isinstance(condition, Variable), \
            "Condition must be a variable if either if_true or if_false is a variable."
        assert isinstance(if_false, Variable) and isinstance(if_false, Variable), \
            "Both if_true and if_false must be variables if either is one."
        assert if_true.data.type() == if_false.data.type(), \
            "Type mismatch: {} and {}".format(if_true.data.type(), if_false.data.type())
    else:
        assert not isinstance(condition, Variable), \
            "Condition must not be a variable because neither if_true nor if_false is one."
        # noinspection PyArgumentList
        assert if_true.type() == if_false.type(), \
            "Type mismatch: {} and {}".format(if_true.data.type(), if_false.data.type())
    casted_condition = condition.type_as(if_true)
    output = casted_condition * if_true + (1 - casted_condition) * if_false
    return output
