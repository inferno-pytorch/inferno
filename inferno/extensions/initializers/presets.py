import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from functools import partial

from .base import Initialization, Initializer


__all__ = ['Constant', 'NormalWeights',
           'SELUWeightsZeroBias',
           'ELUWeightsZeroBias',
           'OrthogonalWeightsZeroBias',
           'KaimingNormalWeightsZeroBias']


class Constant(Initializer):
    """Initialize with a constant."""
    def __init__(self, constant):
        self.constant = constant

    def call_on_tensor(self, tensor):
        if isinstance(tensor, Variable):
            tensor = tensor.data
        tensor.fill_(self.constant)
        return tensor


class NormalWeights(Initializer):
    """
    Initialize weights with random numbers drawn from the normal distribution at
    `mean` and `stddev`.
    """
    def __init__(self, mean=0., stddev=1., sqrt_gain_over_fan_in=None):
        self.mean = mean
        self.stddev = stddev
        self.sqrt_gain_over_fan_in = sqrt_gain_over_fan_in

    def compute_fan_in(self, tensor):
        if tensor.dim() == 2:
            return tensor.size(1)
        else:
            return np.prod(list(tensor.size())[1:])

    def call_on_weight(self, tensor):
        if isinstance(tensor, Variable):
            self.call_on_weight(tensor.data)
            return tensor
        # Compute stddev if required
        if self.sqrt_gain_over_fan_in is not None:
            stddev = self.stddev * \
                     np.sqrt(self.sqrt_gain_over_fan_in / self.compute_fan_in(tensor))
        else:
            stddev = self.stddev
        # Init
        tensor.normal_(self.mean, stddev)


class OrthogonalWeightsZeroBias(Initialization):
    def __init__(self, orthogonal_gain=1.):
        # This prevents a deprecated warning in Pytorch 0.4+
        orthogonal = getattr(init, 'orthogonal_', init.orthogonal)
        super(OrthogonalWeightsZeroBias, self)\
            .__init__(weight_initializer=partial(orthogonal, gain=orthogonal_gain),
                      bias_initializer=Constant(0.))


class KaimingNormalWeightsZeroBias(Initialization):
    def __init__(self, relu_leakage=0):
        # This prevents a deprecated warning in Pytorch 0.4+
        kaiming_normal = getattr(init, 'kaiming_normal_', init.kaiming_normal)
        super(KaimingNormalWeightsZeroBias, self)\
            .__init__(weight_initializer=partial(kaiming_normal, a=relu_leakage),
                      bias_initializer=Constant(0.))


class SELUWeightsZeroBias(Initialization):
    def __init__(self):
        super(SELUWeightsZeroBias, self)\
            .__init__(weight_initializer=NormalWeights(sqrt_gain_over_fan_in=1.),
                      bias_initializer=Constant(0.))


class ELUWeightsZeroBias(Initialization):
    def __init__(self):
        super(ELUWeightsZeroBias, self)\
            .__init__(weight_initializer=NormalWeights(sqrt_gain_over_fan_in=1.5505188080679277),
                      bias_initializer=Constant(0.))

