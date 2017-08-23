import torch.nn.init as init
from torch.autograd import Variable
from functools import partial

from .base import Initialization, Initializer


class Constant(Initializer):
    """Initialize with a constant."""
    def __init__(self, constant):
        self.constant = constant

    def call_on_tensor(self, tensor):
        if isinstance(tensor, Variable):
            self.call_on_tensor(tensor.data)
            return tensor
        tensor.fill_(self.constant)
        return tensor


class OrthogonalWeightsZeroBias(Initialization):
    def __init__(self, orthogonal_gain=1.):
        super(OrthogonalWeightsZeroBias, self)\
            .__init__(weight_initializer=partial(init.orthogonal, gain=orthogonal_gain),
                      bias_initializer=Constant(0.))


class KaimingNormalWeightsZeroBias(Initialization):
    def __init__(self, relu_leakage=0):
        super(KaimingNormalWeightsZeroBias, self)\
            .__init__(weight_initializer=partial(init.kaiming_normal, a=relu_leakage),
                      bias_initializer=Constant(0.))