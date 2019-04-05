import torch.nn.functional as F
import torch.nn as nn
from ...utils.torch_utils import where

__all__ = ['SELU','get_activation']
_all = __all__

class SELU(nn.Module):
    def forward(self, input):
        return self.selu(input)

    @staticmethod
    def selu(x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        # noinspection PyTypeChecker
        return scale * where(x >= 0, x, alpha * F.elu(x))


def get_activation(activation):
    # get the final output and activation activation
    if isinstance(activation, str):
        activation_mod = getattr(nn, final_activation)()
    elif isinstance(activation, nn.Module):
        activation_mod = activation
    elif activation is None:
        activation_mod = None
    else:
        raise NotImplementedError("Activation of type %s is not supported" % type(activation))
    return activation_mod