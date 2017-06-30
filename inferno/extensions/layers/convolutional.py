import torch.nn as nn


class ConvActivation(nn.Module):
    """Convolutional layer with 'SAME' padding followed by an activation."""
    def __init__(self, in_channels, out_channels, kernel_size, dim, activation,
                 stride=1, dilation=1, bias=True):
        super(ConvActivation, self).__init__()
        # Validate dim
        assert dim in [2, 3]
        self.dim = dim
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
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)
        elif isinstance(activation, nn.Module):
            self.activation = activation
        else:
            raise NotImplementedError

    def forward(self, input):
        conved = self.conv(input)
        activated = self.activation(conved)
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
        return padding


class ConvELU2D(ConvActivation):
    """2D Convolutional layer with 'SAME' padding and ELU."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvELU2D, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        dim=2,
                                        activation='ELU')


class ConvELU3D(ConvActivation):
    """3D Convolutional layer with 'SAME' padding and ELU."""
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvELU3D, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        dim=3,
                                        activation='ELU')


class StridedConvELU2D(ConvActivation):
    """2D strided convolutional layer with 'SAME' padding and ELU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(StridedConvELU2D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dim=2,
                                               activation='ELU')


class StridedConvELU3D(ConvActivation):
    """2D strided convolutional layer with 'SAME' padding and ELU."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=2):
        super(StridedConvELU3D, self).__init__(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               dim=3,
                                               activation='ELU')
