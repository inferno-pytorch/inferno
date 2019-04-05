import torch.nn as nn
from ..initializers import OrthogonalWeightsZeroBias, KaimingNormalWeightsZeroBias, \
    SELUWeightsZeroBias
from ..initializers import Initializer
from .activations import SELU
from ...utils.exceptions import assert_, ShapeError


__all__ = ['ConvActivation',
           'ConvELU','ConvELU1D', 'ConvELU2D', 'ConvELU3D',
           'ConvSELU','ConvSELU1D', 'ConvSELU2D', 'ConvSELU3D',
           'ConvReLU','ConvReLU1D', 'ConvReLU2D', 'ConvReLU3D',
           'ConvSigmoid','ConvSigmoid1D','ConvSigmoid2D', 'ConvSigmoid3D',
           'DeconvELU','DeconvELU1D','DeconvELU2D', 'DeconvELU3D',
           'StridedConvELU','StridedConvELU1D', 'StridedConvELU2D', 'StridedConvELU3D',
           'DilatedConvELU','DilatedConvELU1D','DilatedConvELU2D', 'DilatedConvELU3D',
           'Conv','Conv1D', 'Conv2D', 'Conv3D',
           'BNReLUConv','BNReLUConv1D','BNReLUConv2D', 'BNReLUConv3D',
           'BNReLUDepthwiseConv','BNReLUDepthwiseConv1D', 'BNReLUDepthwiseConv2D',
           'BNReLUDilatedConv','BNReLUDilatedConv1D', 'BNReLUDilatedConv2D', 
           'DilatedConv','DilatedConv2D','DilatedConv3D',
           'GlobalConv2D']
_all = __all__


class ConvActivation(nn.Module):
    """Convolutional layer with 'SAME' padding followed by an activation."""
    def __init__(self, in_channels, out_channels, kernel_size, dim, activation,
                 stride=1, dilation=1, groups=None, depthwise=False, bias=True,
                 deconv=False, initialization=None):
        super(ConvActivation, self).__init__()
        # Validate dim
        assert_(dim in [1, 2, 3], "`dim` must be one of [1, 2, 3], got {}.".format(dim), ShapeError)
        self.dim = dim
        # Check if depthwise
        if depthwise:
            assert_(in_channels == out_channels,
                    "For depthwise convolutions, number of input channels (given: {}) "
                    "must equal the number of output channels (given {})."
                    .format(in_channels, out_channels),
                    ValueError)
            assert_(groups is None or groups == in_channels,
                    "For depthwise convolutions, groups (given: {}) must "
                    "equal the number of channels (given: {}).".format(groups, in_channels))
            groups = in_channels
        else:
            groups = 1 if groups is None else groups
        self.depthwise = depthwise
        if not deconv:
            # Get padding
            padding = self.get_padding(kernel_size, dilation)
            self.conv = getattr(nn, 'Conv{}d'.format(self.dim))(in_channels=in_channels,
                                                                out_channels=out_channels,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                stride=stride,
                                                                dilation=dilation,
                                                                groups=groups,
                                                                bias=bias)
        else:
            self.conv = getattr(nn, 'ConvTranspose{}d'.format(self.dim))(in_channels=in_channels,
                                                                         out_channels=out_channels,
                                                                         kernel_size=kernel_size,
                                                                         stride=stride,
                                                                         dilation=dilation,
                                                                         groups=groups,
                                                                         bias=bias)
        if initialization is None:
            pass
        elif isinstance(initialization, Initializer):
            self.conv.apply(initialization)
        else:
            raise NotImplementedError

        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        elif isinstance(activation, nn.Module):
            self.activation = activation
        elif activation is None:
            self.activation = None
        else:
            raise NotImplementedError

    def forward(self, input):
        conved = self.conv(input)
        if self.activation is not None:
            activated = self.activation(conved)
        else:
            # No activation
            activated = conved
        return activated

    def _pair_or_triplet(self, object_):
        if isinstance(object_, (list, tuple)):
            assert len(object_) == self.dim
            return object_
        else:
            object_ = [object_] * self.dim
            return object_

    def _get_padding(self, _kernel_size, _dilation):
        assert isinstance(_kernel_size, int)
        assert isinstance(_dilation, int)
        assert _kernel_size % 2 == 1
        return ((_kernel_size - 1) // 2) * _dilation

    def get_padding(self, kernel_size, dilation):
        kernel_size = self._pair_or_triplet(kernel_size)
        dilation = self._pair_or_triplet(dilation)
        padding = [self._get_padding(_kernel_size, _dilation)
                   for _kernel_size, _dilation in zip(kernel_size, dilation)]
        return tuple(padding)

class ConvReLU(ConvActivation):
    """Convolutional layer with 'SAME' padding, ReLU and Kaiming normal weight initialization."""
    def __init__(self, dim, in_channels, out_channels, kernel_size):
        super(ConvReLU, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=dim,
                                         activation='ReLU',
                                         initialization=KaimingNormalWeightsZeroBias())

class ConvReLU1D(ConvActivation):
    """1D Convolutional layer with 'SAME' padding, ReLU and Kaiming normal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvReLU1D, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=1,
                                         activation='ReLU',
                                         initialization=KaimingNormalWeightsZeroBias())

class ConvReLU2D(ConvActivation):
    """2D Convolutional layer with 'SAME' padding, ReLU and Kaiming normal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvReLU2D, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=2,
                                         activation='ReLU',
                                         initialization=KaimingNormalWeightsZeroBias())

class ConvReLU3D(ConvActivation):
    """3D Convolutional layer with 'SAME' padding, ReLU and Kaiming normal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvReLU3D, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=3,
                                         activation='ReLU',
                                         initialization=KaimingNormalWeightsZeroBias())

class ConvELU(ConvActivation):
    """1D Convolutional layer with 'SAME' padding, ELU and orthogonal weight initialization."""
    def __init__(self, dim, in_channels, out_channels, kernel_size):
        super(ConvELU, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        dim=dim,
                                        activation='ELU',
                                        initialization=OrthogonalWeightsZeroBias())

class ConvELU1D(ConvActivation):
    """1D Convolutional layer with 'SAME' padding, ELU and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvELU1D, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        dim=1,
                                        activation='ELU',
                                        initialization=OrthogonalWeightsZeroBias())

class ConvELU2D(ConvActivation):
    """2D Convolutional layer with 'SAME' padding, ELU and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvELU2D, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        dim=2,
                                        activation='ELU',
                                        initialization=OrthogonalWeightsZeroBias())

class ConvELU3D(ConvActivation):
    """3D Convolutional layer with 'SAME' padding, ELU and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvELU3D, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        dim=3,
                                        activation='ELU',
                                        initialization=OrthogonalWeightsZeroBias())

class ConvSELU(ConvActivation):
    """Convolutional layer with SELU activation and the appropriate weight initialization."""
    def __init__(self, dim, in_channels, out_channels, kernel_size):
        if hasattr(nn, 'SELU'):
            # Pytorch 0.2: Use built in SELU
            activation = nn.SELU(inplace=True)
        else:
            # Pytorch < 0.1.12: Use handmade SELU
            activation = SELU()
        super(ConvSELU, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=dim,
                                         activation=activation,
                                         initialization=SELUWeightsZeroBias())


class ConvSELU1D(ConvActivation):
    """1D Convolutional layer with SELU activation and the appropriate weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        if hasattr(nn, 'SELU'):
            # Pytorch 0.2: Use built in SELU
            activation = nn.SELU(inplace=True)
        else:
            # Pytorch < 0.1.12: Use handmade SELU
            activation = SELU()
        super(ConvSELU1D, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=1,
                                         activation=activation,
                                         initialization=SELUWeightsZeroBias())

class ConvSELU2D(ConvActivation):
    """2D Convolutional layer with SELU activation and the appropriate weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        if hasattr(nn, 'SELU'):
            # Pytorch 0.2: Use built in SELU
            activation = nn.SELU(inplace=True)
        else:
            # Pytorch < 0.1.12: Use handmade SELU
            activation = SELU()
        super(ConvSELU2D, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=2,
                                         activation=activation,
                                         initialization=SELUWeightsZeroBias())

class ConvSELU3D(ConvActivation):
    """3D Convolutional layer with SELU activation and the appropriate weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        if hasattr(nn, 'SELU'):
            # Pytorch 0.2: Use built in SELU
            activation = nn.SELU(inplace=True)
        else:
            # Pytorch < 0.1.12: Use handmade SELU
            activation = SELU()
        super(ConvSELU3D, self).__init__(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         dim=3,
                                         activation=activation,
                                         initialization=SELUWeightsZeroBias())

class ConvSigmoid(ConvActivation):
    """Convolutional layer with 'SAME' padding, Sigmoid and orthogonal weight initialization."""
    def __init__(self, dim, in_channels, out_channels, kernel_size):
        super(ConvSigmoid, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            dim=dim,
                                            activation='Sigmoid',
                                            initialization=OrthogonalWeightsZeroBias())

class ConvSigmoid1D(ConvActivation):
    """1D Convolutional layer with 'SAME' padding, Sigmoid and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvSigmoid1D, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            dim=1,
                                            activation='Sigmoid',
                                            initialization=OrthogonalWeightsZeroBias())

class ConvSigmoid2D(ConvActivation):
    """2D Convolutional layer with 'SAME' padding, Sigmoid and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvSigmoid2D, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            dim=2,
                                            activation='Sigmoid',
                                            initialization=OrthogonalWeightsZeroBias())

class ConvSigmoid3D(ConvActivation):
    """3D Convolutional layer with 'SAME' padding, Sigmoid and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvSigmoid3D, self).__init__(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            dim=3,
                                            activation='Sigmoid',
                                            initialization=OrthogonalWeightsZeroBias())

class DeconvELU(ConvActivation):
    """deconvolutional layer with ELU and orthogonal weight initialization."""
    def __init__(self, dim, in_channels, out_channels, kernel_size=2):
        super(DeconvELU, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          dim=dim,
                                          activation='ELU',
                                          deconv=True,
                                          initialization=OrthogonalWeightsZeroBias())

class DeconvELU1D(ConvActivation):
    """1D deconvolutional layer with ELU and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(DeconvELU1D, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          dim=1,
                                          activation='ELU',
                                          deconv=True,
                                          initialization=OrthogonalWeightsZeroBias())

class DeconvELU2D(ConvActivation):
    """2D deconvolutional layer with ELU and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(DeconvELU2D, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          dim=2,
                                          activation='ELU',
                                          deconv=True,
                                          initialization=OrthogonalWeightsZeroBias())

class DeconvELU3D(ConvActivation):
    """3D deconvolutional layer with ELU and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(DeconvELU3D, self).__init__(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          dim=3,
                                          activation='ELU',
                                          deconv=True,
                                          initialization=OrthogonalWeightsZeroBias())

class StridedConvELU(ConvActivation):
    """
    strided convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=2):
        super(StridedConvELU, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dim=dim,
                                               activation='ELU',
                                               initialization=OrthogonalWeightsZeroBias())

class StridedConvELU1D(ConvActivation):
    """
    1D strided convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self , in_channels, out_channels, kernel_size, stride=2):
        super(StridedConvELU1D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dim=1,
                                               activation='ELU',
                                               initialization=OrthogonalWeightsZeroBias())

class StridedConvELU2D(ConvActivation):
    """
    2D strided convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(StridedConvELU2D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dim=2,
                                               activation='ELU',
                                               initialization=OrthogonalWeightsZeroBias())

class StridedConvELU3D(ConvActivation):
    """
    2D strided convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(StridedConvELU3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dim=3,
                                               activation='ELU',

                                               initialization=OrthogonalWeightsZeroBias())
class DilatedConvELU(ConvActivation):
    """
    dilated convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConvELU, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=dim,
                                               activation='ELU',
                                               initialization=OrthogonalWeightsZeroBias())

class DilatedConvELU1D(ConvActivation):
    """
    1D dilated convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConvELU1D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=1,
                                               activation='ELU',
                                               initialization=OrthogonalWeightsZeroBias())

class DilatedConvELU2D(ConvActivation):
    """
    2D dilated convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConvELU2D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=2,
                                               activation='ELU',
                                               initialization=OrthogonalWeightsZeroBias())

class DilatedConvELU3D(ConvActivation):
    """
    3D dilated convolutional layer with 'SAME' padding, ELU and orthogonal
    weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConvELU3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=3,
                                               activation='ELU',
                                               initialization=OrthogonalWeightsZeroBias())
class DilatedConv(ConvActivation):
    """dilated convolutional layer with 'SAME' padding, no activation and orthogonal weight initialization."""
    def __init__(self, dim, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConv, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=dim,
                                               activation=None,
                                               initialization=OrthogonalWeightsZeroBias())

class DilatedConv1D(ConvActivation):
    """1D dilated convolutional layer with 'SAME' padding, no activation and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConv1D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=1,
                                               activation=None,
                                               initialization=OrthogonalWeightsZeroBias())

class DilatedConv2D(ConvActivation):
    """2D dilated convolutional layer with 'SAME' padding, no activation and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConv2D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=2,
                                               activation=None,
                                               initialization=OrthogonalWeightsZeroBias())

class DilatedConv3D(ConvActivation):
    """3D dilated convolutional layer with 'SAME' padding, no activation and orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(DilatedConv3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=3,
                                               activation=None,
                                               initialization=OrthogonalWeightsZeroBias())

class Conv(ConvActivation):
    """
    convolutional layer with same padding and orthogonal weight initialization.
    By default, this layer does not apply an activation function.
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, dilation=1, stride=1,
                 activation=None):
        super(Conv, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     stride=stride,
                                     dim=dim,
                                     activation=activation,
                                     initialization=OrthogonalWeightsZeroBias())

class Conv1D(ConvActivation):
    """
    1D convolutional layer with same padding and orthogonal weight initialization.
    By default, this layer does not apply an activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1,
                 activation=None):
        super(Conv1D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     stride=stride,
                                     dim=1,
                                     activation=activation,
                                     initialization=OrthogonalWeightsZeroBias())

class Conv2D(ConvActivation):
    """
    2D convolutional layer with same padding and orthogonal weight initialization.
    By default, this layer does not apply an activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1,
                 activation=None):
        super(Conv2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     stride=stride,
                                     dim=2,
                                     activation=activation,
                                     initialization=OrthogonalWeightsZeroBias())

class Conv3D(ConvActivation):
    """
    3D convolutional layer with same padding and orthogonal weight initialization.
    By default, this layer does not apply an activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1,
                 activation=None):
        super(Conv3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     stride=stride,
                                     dim=3,
                                     activation=activation,
                                     initialization=OrthogonalWeightsZeroBias())

class Deconv(ConvActivation):
    """deconvolutional layer with orthogonal weight initialization."""
    def __init__(self, dim, in_channels, out_channels, kernel_size=2, stride=2):
        super(Deconv, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       dim=dim,
                                       stride=stride,
                                       activation=None,
                                       deconv=True,
                                       initialization=OrthogonalWeightsZeroBias())
class Deconv1D(ConvActivation):
    """1D deconvolutional layer with orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Deconv1D, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       dim=1,
                                       stride=stride,
                                       activation=None,
                                       deconv=True,
                                       initialization=OrthogonalWeightsZeroBias())

class Deconv2D(ConvActivation):
    """2D deconvolutional layer with orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Deconv2D, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       dim=2,
                                       stride=stride,
                                       activation=None,
                                       deconv=True,
                                       initialization=OrthogonalWeightsZeroBias())

class Deconv3D(ConvActivation):
    """2D deconvolutional layer with orthogonal weight initialization."""
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super(Deconv3D, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=kernel_size,
                                       dim=3,
                                       stride=stride,
                                       activation=None,
                                       deconv=True,
                                       initialization=OrthogonalWeightsZeroBias())

# noinspection PyUnresolvedReferences
class _BNReLUSomeConv(object):
    def forward(self, input):
        normed = self.batchnorm(input)
        activated = self.activation(normed)
        conved = self.conv(activated)
        return conved

class BNReLUConv(_BNReLUSomeConv, ConvActivation):
    """
    BN-ReLU-Conv layer with 'SAME' padding and He weight initialization.
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=1):
        super(BNReLUConv, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           dim=dim,
                                           stride=stride,
                                           activation=nn.ReLU(inplace=True),
                                           initialization=KaimingNormalWeightsZeroBias(0))
        BatchNormNd = getattr(nn, 'BatchNorm{0}d'.format(int(dim)))
        self.batchnorm = BatchNormNd(in_channels)

class BNReLUConv1D(_BNReLUSomeConv, ConvActivation):
    """
    1D BN-ReLU-Conv layer with 'SAME' padding and He weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BNReLUConv1D, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           dim=1,
                                           stride=stride,
                                           activation=nn.ReLU(inplace=True),
                                           initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm1d(in_channels)

class BNReLUConv2D(_BNReLUSomeConv, ConvActivation):
    """
    2D BN-ReLU-Conv layer with 'SAME' padding and He weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BNReLUConv2D, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           dim=2,
                                           stride=stride,
                                           activation=nn.ReLU(inplace=True),
                                           initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm2d(in_channels)

class BNReLUConv3D(_BNReLUSomeConv, ConvActivation):
    """
    3D BN-ReLU-Conv layer with 'SAME' padding and He weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BNReLUConv3D, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           dim=3,
                                           stride=stride,
                                           activation=nn.ReLU(inplace=True),
                                           initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm3d(in_channels)

class BNReLUDilatedConv(_BNReLUSomeConv,ConvActivation):
    """
    dilated convolutional layer with 'SAME' padding, Batch norm,  Relu and He
    weight initialization.
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, dilation=1):
        super(BNReLUDilatedConv1D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=dim,
                                               activation=nn.ReLU(inplace=True),
                                               initialization=KaimingNormalWeightsZeroBias(0))
        BatchNormNd = getattr(nn, 'BatchNorm{0}d'.format(int(dim)))
        self.batchnorm = BatchNormNd(in_channels)

class BNReLUDilatedConv1D(_BNReLUSomeConv,ConvActivation):
    """
    1D dilated convolutional layer with 'SAME' padding, Batch norm,  Relu and He
    weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(BNReLUDilatedConv1D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=1,
                                               activation=nn.ReLU(inplace=True),
                                               initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm2d(in_channels)

class BNReLUDilatedConv2D(_BNReLUSomeConv,ConvActivation):
    """
    2D dilated convolutional layer with 'SAME' padding, Batch norm,  Relu and He
    weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=2):
        super(BNReLUDilatedConv2D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               dilation=dilation,
                                               dim=2,
                                               activation=nn.ReLU(inplace=True),
                                               initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm2d(in_channels)

class BNReLUDeconv(_BNReLUSomeConv, ConvActivation):
    """
    BN-ReLU-Deconv layer with He weight initialization and (default) stride 1.
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size, stride=2):
        super(BNReLUDeconv, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             dim=dim,
                                             stride=stride,
                                             deconv=True,
                                             activation=nn.ReLU(inplace=True),
                                             initialization=KaimingNormalWeightsZeroBias(0))
        BatchNormNd = getattr(nn, 'BatchNorm{0}d'.format(int(dim)))
        self.batchnorm = BatchNormNd(in_channels)

class BNReLUDeconv1D(_BNReLUSomeConv, ConvActivation):
    """
    1D BN-ReLU-Deconv layer with He weight initialization and (default) stride 1.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(BNReLUDeconv1D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             dim=1,
                                             stride=stride,
                                             deconv=True,
                                             activation=nn.ReLU(inplace=True),
                                             initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm1d(in_channels)

class BNReLUDeconv2D(_BNReLUSomeConv, ConvActivation):
    """
    2D BN-ReLU-Deconv layer with He weight initialization and (default) stride 2.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(BNReLUDeconv2D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             dim=2,
                                             stride=stride,
                                             deconv=True,
                                             activation=nn.ReLU(inplace=True),
                                             initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm2d(in_channels)

class BNReLUDeconv3D(_BNReLUSomeConv, ConvActivation):
    """
    3D BN-ReLU-Deconv layer with He weight initialization and (default) stride 2.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(BNReLUDeconv3D, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             dim=3,
                                             stride=stride,
                                             deconv=True,
                                             activation=nn.ReLU(inplace=True),
                                             initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm2d(in_channels)

class BNReLUDepthwiseConv(_BNReLUSomeConv, ConvActivation):
    """
    1D BN-ReLU-Conv layer with 'SAME' padding, He weight initialization and depthwise convolution.
    Note that depthwise convolutions require `in_channels == out_channels`.
    """
    def __init__(self, dim, in_channels, out_channels, kernel_size):
        # We know that in_channels == out_channels, but we also want a consistent API.
        # As a compromise, we allow that out_channels be None or 'auto'.
        out_channels = in_channels if out_channels in [None, 'auto'] else out_channels
        super(BNReLUDepthwiseConv, self).__init__(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    dim=dim,
                                                    depthwise=True,
                                                    activation=nn.ReLU(inplace=True),
                                                    initialization=KaimingNormalWeightsZeroBias(0))
        BatchNormNd = getattr(nn, 'BatchNorm{0}d'.format(int(dim)))
        self.batchnorm = BatchNormNd(in_channels)


class BNReLUDepthwiseConv1D(_BNReLUSomeConv, ConvActivation):
    """
    1D BN-ReLU-Conv layer with 'SAME' padding, He weight initialization and depthwise convolution.
    Note that depthwise convolutions require `in_channels == out_channels`.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        # We know that in_channels == out_channels, but we also want a consistent API.
        # As a compromise, we allow that out_channels be None or 'auto'.
        out_channels = in_channels if out_channels in [None, 'auto'] else out_channels
        super(BNReLUDepthwiseConv1D, self).__init__(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    dim=1,
                                                    depthwise=True,
                                                    activation=nn.ReLU(inplace=True),
                                                    initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm1d(in_channels)

class BNReLUDepthwiseConv2D(_BNReLUSomeConv, ConvActivation):
    """
    2D BN-ReLU-Conv layer with 'SAME' padding, He weight initialization and depthwise convolution.
    Note that depthwise convolutions require `in_channels == out_channels`.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        # We know that in_channels == out_channels, but we also want a consistent API.
        # As a compromise, we allow that out_channels be None or 'auto'.
        out_channels = in_channels if out_channels in [None, 'auto'] else out_channels
        super(BNReLUDepthwiseConv2D, self).__init__(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    dim=2,
                                                    depthwise=True,
                                                    activation=nn.ReLU(inplace=True),
                                                    initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm2d(in_channels)

class GlobalConv2D(nn.Module):
    """From https://arxiv.org/pdf/1703.02719.pdf
    Main idea: we can have a bigger kernel size computationally acceptable
    if we separate 2D-conv in 2 1D-convs """
    def __init__(self, in_channels, out_channels, kernel_size, local_conv_type,
                 activation=None, use_BN=False, **kwargs):
        super(GlobalConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
           kernel_size = (kernel_size,)*2
        self.kwargs=kwargs
        self.conv1a = local_conv_type(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=(kernel_size[0], 1), **kwargs)
        self.conv1b = local_conv_type(in_channels=self.out_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=(1, kernel_size[1]), **kwargs)
        self.conv2a = local_conv_type(in_channels=self.in_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=(1, kernel_size[1]), **kwargs)
        self.conv2b = local_conv_type(in_channels=self.out_channels,
                                      out_channels=self.out_channels,
                                      kernel_size=(kernel_size[0], 1), **kwargs)
        if use_BN:
            self.batchnorm = nn.BatchNorm2d(self.out_channels)
        else:
            self.batchnorm = None
        self.activation = activation

    def forward(self, input_):
        out1 = self.conv1a(input_)
        out1 = self.conv1b(out1)
        out2 = self.conv2a(input_)
        out2 = self.conv2b(out2)
        out = out1.add(1,out2)
        if self.activation is not None:
            out = self.activation(out)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        return out