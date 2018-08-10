import torch.nn.functional as F
import torch.nn as nn
from ...utils.torch_utils import where

__all__ = ['SELU']
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