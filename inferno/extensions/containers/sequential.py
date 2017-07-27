import torch.nn as nn
from ...utils import python_utils as pyu


class Sequential2(nn.Sequential):
    """Another sequential container.
    Identitcal to torch.nn.Sequential, except that modules may return multiple outputs and
    accept multiple inputs.
    """
    def forward(self, input):
        for module in self._modules.values():
            input = module(*pyu.to_iterable(input))
        return pyu.from_iterable(input)

    def __len__(self):
        return len(self._modules.values())

