import numpy as np
import torch

from .python_utils import delayed_keyboard_interrupt
from .exceptions import assert_, ShapeError, NotUnwrappableError


def unwrap(input_, to_cpu=True, as_numpy=False, extract_item=False):
    if isinstance(input_, (list, tuple)):
        return type(input_)([unwrap(_t, to_cpu=to_cpu, as_numpy=as_numpy)
                             for _t in input_])
    elif torch.is_tensor(input_):
        tensor = input_
    elif isinstance(input_, np.ndarray):
        return input_
    elif isinstance(input_, (float, int)):
        return input_
    else:
        raise NotUnwrappableError("Cannot unwrap a '{}'."
                                  .format(type(input_).__name__))
    # Transfer to CPU if required
    if to_cpu:
        with delayed_keyboard_interrupt():
            tensor = tensor.cpu().detach()
    # Convert to numpy if required
    if as_numpy:
        return tensor.cpu().detach().numpy()
    elif extract_item:
        try:
            return tensor.item()
        except AttributeError:
            return tensor[0]
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
    condition : torch.ByteTensor or torch.cuda.ByteTensor
        Condition to check.
    if_true : torch.Tensor or torch.cuda.Tensor
        Output value if condition is true.
    if_false: torch.Tensor or torch.cuda.Tensor
        Output value if condition is false

    Returns
    -------
    torch.Tensor

    Raises
    ------
    AssertionError
        if if_true and if_false don't have the same datatype.
    """
    # noinspection PyArgumentList
    assert if_true.type() == if_false.type(), \
        "Type mismatch: {} and {}".format(if_true.data.type(), if_false.data.type())
    casted_condition = condition.type_as(if_true)
    output = casted_condition * if_true + (1 - casted_condition) * if_false
    return output


def flatten_samples(input_):
    """
    Flattens a tensor or a variable such that the channel axis is first and the sample axis
    is second. The shapes are transformed as follows:
        (N, C, H, W) --> (C, N * H * W)
        (N, C, D, H, W) --> (C, N * D * H * W)
        (N, C) --> (C, N)
    The input must be atleast 2d.
    """
    assert_(input_.dim() >= 2,
            "Tensor or variable must be atleast 2D. Got one of dim {}."
            .format(input_.dim()),
            ShapeError)
    # Get number of channels
    num_channels = input_.size(1)
    # Permute the channel axis to first
    permute_axes = list(range(input_.dim()))
    permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
    # For input shape (say) NCHW, this should have the shape CNHW
    permuted = input_.permute(*permute_axes).contiguous()
    # Now flatten out all but the first axis and return
    flattened = permuted.view(num_channels, -1)
    return flattened


def clip_gradients_(parameters, mode, norm_or_value):
    assert_(mode in ['norm', 'value'],
            f"Mode must be 'norm' or 'value', got '{mode}' instead.",
            ValueError)
    if mode == 'norm':
        torch.nn.utils.clip_grad_norm_(parameters, norm_or_value)
    elif mode == 'value':
        torch.nn.utils.clip_grad_value_(parameters, norm_or_value)
    else:
        raise NotImplementedError
