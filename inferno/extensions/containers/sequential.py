import torch.nn as nn
from ...utils import python_utils as pyu


__all__ = ['Sequential1', 'Sequential2']


class Sequential1(nn.Sequential):
    """Like torch.nn.Sequential, but with a few extra methods."""
    def __len__(self):
        return len(self._modules.values())


class Sequential2(Sequential1):
    """Another sequential container.
    Identitcal to torch.nn.Sequential, except that modules may return multiple outputs and
    accept multiple inputs.
    """
    def forward(self, *input):
        for module in self._modules.values():
            input = pyu.to_iterable(module(*pyu.to_iterable(input)))
        return pyu.from_iterable(input)
