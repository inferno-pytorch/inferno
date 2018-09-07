import torch.nn as nn
from .convolutional import BNReLUConv2D, BNReLUDeconv2D, Conv2D, Deconv2D
from ...utils import python_utils as pyu
from ...utils.exceptions import assert_

__all__ = ['ResidualBlock', 'PreActSimpleResidualBlock']
_all = __all__


class ResidualBlock(nn.Module):
    def __init__(self, layers, resample=None):
        super(ResidualBlock, self).__init__()
        assert pyu.is_listlike(layers)
        self.layers = nn.Sequential(*layers)
        self.resample = resample

    def forward(self, input):
        preaddition = self.layers(input)
        if self.resample is not None:
            skip = self.resample(input)
        else:
            skip = input
        output = preaddition + skip
        return output


class PreActSimpleResidualBlock(ResidualBlock):
    def __init__(self, in_channels, num_hidden_channels, upsample=False, downsample=False):
        layers = []
        if downsample:
            assert_(not upsample, "Both downsample and upsample is set to true.", ValueError)
            layers.append(BNReLUConv2D(in_channels=in_channels,
                                       out_channels=num_hidden_channels,
                                       kernel_size=3,
                                       stride=2))
            resample = nn.Sequential(Conv2D(in_channels=in_channels,
                                            out_channels=in_channels,
                                            kernel_size=1, stride=2),
                                     nn.BatchNorm2d(in_channels))
        elif upsample:
            layers.append(BNReLUDeconv2D(in_channels=in_channels,
                                         out_channels=num_hidden_channels,
                                         kernel_size=2,
                                         stride=2))
            resample = nn.Sequential(Deconv2D(in_channels=in_channels,
                                              out_channels=in_channels,
                                              kernel_size=2, stride=2),
                                     nn.BatchNorm2d(in_channels))
        else:
            layers.append(BNReLUConv2D(in_channels=in_channels,
                                       out_channels=num_hidden_channels,
                                       kernel_size=3))
            resample = None
        layers.append(BNReLUConv2D(in_channels=num_hidden_channels,
                                   out_channels=in_channels,
                                   kernel_size=3))
        super(PreActSimpleResidualBlock, self).__init__(layers, resample)


# TODO PreActBottleneckResidualBlock
