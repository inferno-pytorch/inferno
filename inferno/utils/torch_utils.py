import numpy as np
import torch
from torch.autograd import Variable

from .python_utils import delayed_keyboard_interrupt
from .exceptions import assert_, ShapeError, NotUnwrappableError


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
        raise NotUnwrappableError("Cannot unwrap a '{}'."
                                  .format(type(tensor_or_variable).__name__))
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
    missed_tensor_classes = (torch.HalfTensor,)
    return torch.is_tensor(object_) or isinstance(object_, missed_tensor_classes)


def is_label_tensor(object_):
    return is_tensor(object_) and object_.type() in ['torch.LongTensor', 'torch.cuda.LongTensor']


def is_image_tensor(object_):
    return is_tensor(object_) and object_.dim() == 4


def is_volume_tensor(object_):
    return is_tensor(object_) and object_.dim() == 5


def is_image_or_volume_tensor(object_):
    return is_image_tensor(object_) or is_volume_tensor(object_)


def is_label_image_tensor(object_):
    return is_label_tensor(object_) and object_.dim() == 3


def is_label_volume_tensor(object_):
    return is_label_tensor(object_) and object_.dim() == 4


def is_label_image_or_volume_tensor(object_):
    return is_label_image_tensor(object_) or is_label_volume_tensor(object_)


def is_matrix_tensor(object_):
    return is_tensor(object_) and object_.dim() == 2


def is_scalar_tensor(object_):
    return is_tensor(object_) and object_.dim() <= 1 and object_.numel() == 1


def is_vector_tensor(object_):
    return is_tensor(object_) and object_.dim() == 1 and object_.numel() > 1


def assert_same_size(tensor_1, tensor_2):
    assert_(list(tensor_1.size()) == list(tensor_2.size()),
            "Tensor sizes {} and {} do not match.".format(tensor_1.size(), tensor_2.size()),
            ShapeError)


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


def flatten_samples(tensor_or_variable):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    assert_(tensor_or_variable.dim() >= 2,
            "Tensor or variable must be atleast 2D. Got one of dim {}."
            .format(tensor_or_variable.dim()),
            ShapeError)
    # Get number of channels
    num_channels = tensor_or_variable.size(1)
    # Permute the channel axis to first
    permute_axes = list(range(tensor_or_variable.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    # For input shape (say) NCHW, this should have the shape CNHW
    permuted = tensor_or_variable.permute(*permute_axes).contiguous()
    # Now flatten out all but the first axis and return
    flattened = permuted.view(num_channels, -1)
    return flattened
