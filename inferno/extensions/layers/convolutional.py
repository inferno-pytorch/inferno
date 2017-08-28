import torch.nn as nn
from ..initializers import OrthogonalWeightsZeroBias, KaimingNormalWeightsZeroBias
from ..initializers import Initializer


class ConvActivation(nn.Module):
    """Convolutional layer with 'SAME' padding followed by an activation."""
    def __init__(self, in_channels, out_channels, kernel_size, dim, activation,
                 stride=1, dilation=1, bias=True, deconv=False, initialization=None):
        super(ConvActivation, self).__init__()
        # Validate dim
        assert dim in [2, 3]
        self.dim = dim
        if not deconv:
            # Get padding
            padding = self.get_padding(kernel_size, dilation)
            # Get convlayer
            self.conv = getattr(nn, 'Conv{}d'.format(self.dim))(in_channels=in_channels,
                                                                out_channels=out_channels,
                                                                kernel_size=kernel_size,
                                                                padding=padding,
                                                                stride=stride,
                                                                dilation=dilation,
                                                                bias=bias)
        else:
            self.conv = getattr(nn, 'ConvTranspose{}d'.format(self.dim))(in_channels=in_channels,
                                                                         out_channels=out_channels,
                                                                         kernel_size=kernel_size,
                                                                         stride=stride,
                                                                         dilation=dilation,
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


class Conv2D(ConvActivation):
    """
    2D convolutional layer with same padding and orthogonal weight initialization.
    This layer does not apply an activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super(Conv2D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     dim=2,
                                     activation=activation,
                                     initialization=OrthogonalWeightsZeroBias())


class Conv3D(ConvActivation):
    """
    3D convolutional layer with same padding and orthogonal weight initialization.
    This layer does not apply an activation function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation=None):
        super(Conv3D, self).__init__(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     dim=3,
                                     activation=activation,
                                     initialization=OrthogonalWeightsZeroBias())


class BNReLUConv2D(ConvActivation):
    """
    2D BN-ReLU-Conv layer with 'SAME' padding and He weight initialization.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BNReLUConv2D, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           dim=2,
                                           activation=nn.ReLU(inplace=True),
                                           initialization=KaimingNormalWeightsZeroBias(0))
        self.batchnorm = nn.BatchNorm2d(in_channels)

    def forward(self, input):
        normed = self.batchnorm(input)
        activated = self.activation(normed)
        conved = self.conv(activated)
        return conved
