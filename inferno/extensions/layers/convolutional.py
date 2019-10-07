import torch.nn as nn
import sys
import functools
from ..initializers import (
    OrthogonalWeightsZeroBias,
    KaimingNormalWeightsZeroBias,
    SELUWeightsZeroBias,
)
from ..initializers import Initializer
from .normalization import BatchNormND
from .activations import SELU
from ...utils.exceptions import assert_, ShapeError
from ...utils.partial_cls import register_partial_cls

# we append to this later on
__all__ = [
    "GlobalConv2D",
]
_all = __all__

register_partial_cls_here = functools.partial(register_partial_cls, module=__name__)


class ConvActivation(nn.Module):
    """Convolutional layer with 'SAME' padding by default followed by an activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dim,
        activation,
        stride=1,
        dilation=1,
        groups=None,
        depthwise=False,
        bias=True,
        deconv=False,
        initialization=None,
        valid_conv=False,
    ):
        super(ConvActivation, self).__init__()
        # Validate dim
        assert_(
            dim in [1, 2, 3],
            "`dim` must be one of [1, 2, 3], got {}.".format(dim),
            ShapeError,
        )
        self.dim = dim
        # Check if depthwise
        if depthwise:

            # We know that in_channels == out_channels, but we also want a consistent API.
            # As a compromise, we allow that out_channels be None or 'auto'.
            out_channels = in_channels if out_channels in [None, "auto"] else out_channel
            assert_(
                in_channels == out_channels,
                "For depthwise convolutions, number of input channels (given: {}) "
                "must equal the number of output channels (given {}).".format(
                    in_channels, out_channels
                ),
                ValueError,
            )
            assert_(
                groups is None or groups == in_channels,
                "For depthwise convolutions, groups (given: {}) must "
                "equal the number of channels (given: {}).".format(groups, in_channels),
            )
            groups = in_channels
        else:
            groups = 1 if groups is None else groups
        self.depthwise = depthwise
        if valid_conv:
            self.conv = getattr(nn, "Conv{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        elif not deconv:
            # Get padding
            padding = self.get_padding(kernel_size, dilation)
            self.conv = getattr(nn, "Conv{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        else:
            self.conv = getattr(nn, "ConvTranspose{}d".format(self.dim))(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
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
        padding = [
            self._get_padding(_kernel_size, _dilation)
            for _kernel_size, _dilation in zip(kernel_size, dilation)
        ]
        return tuple(padding)

# for consistency
ConvActivationND = ConvActivation


# noinspection PyUnresolvedReferences
class _BNReLUSomeConv(object):
    def forward(self, input):
        normed = self.batchnorm(input)
        activated = self.activation(normed)
        conved = self.conv(activated)
        return conved

class BNReLUConvBaseND(_BNReLUSomeConv, ConvActivation):
    def __init__(self, in_channels, out_channels, kernel_size, dim, stride=1, dilation=1, deconv=False):

        super(BNReLUConvBaseND, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=dim,
            stride=stride,
            activation=nn.ReLU(inplace=True),
            dilation=dilation,
            deconv=deconv,
            initialization=KaimingNormalWeightsZeroBias(0),
        )
        self.batchnorm = BatchNormND(dim, in_channels)


def _register_conv_cls(conv_name,  fix=None, default=None):
    if fix is None:
        fix = {}
    if default is None:
        default = {}

    # simple conv activation
    activations = ["ReLU", "ELU", "Sigmoid", "SELU", ""]
    init_map = {
        "ReLU": KaimingNormalWeightsZeroBias,
        "SELU": SELUWeightsZeroBias
    }
    for activation_str in activations:
        cls_name = cls_name = "{}{}ND".format(conv_name,activation_str)
        __all__.append(cls_name)
        initialization_cls = init_map.get(activation_str, OrthogonalWeightsZeroBias)
        if activation_str == "":
            activation = None
            _fix = {**fix}
            _default = {'activation':None}
        elif activation_str == "SELU":
            activation = nn.SELU(inplace=True)
            _fix={**fix, 'activation':activation}
            _default = {**default}
        else:
            activation = activation_str
            _fix={**fix, 'activation':activation}
            _default = {**default}

        register_partial_cls_here(ConvActivation, cls_name,
            fix=_fix,
            default={**_default, 'initialization':initialization_cls()}
        )
        for dim in [1, 2, 3]:
            cls_name = "{}{}{}D".format(conv_name,activation_str, dim)
            __all__.append(cls_name)
            register_partial_cls_here(ConvActivation, cls_name,
                fix={**_fix, 'dim':dim},
                default={**_default, 'initialization':initialization_cls()}
            )

def _register_bnr_conv_cls(conv_name,  fix=None, default=None):
    if fix is None:
        fix = {}
    if default is None:
        default = {}
    for dim in [1, 2, 3]:

        cls_name = "BNReLU{}ND".format(conv_name)
        __all__.append(cls_name)
        register_partial_cls_here(BNReLUConvBaseND, cls_name,fix=fix,default=default)

        for dim in [1, 2, 3]:
            cls_name = "BNReLU{}{}D".format(conv_name, dim)
            __all__.append(cls_name)
            register_partial_cls_here(BNReLUConvBaseND, cls_name,
                fix={**fix, 'dim':dim},
                default=default)

# conv classes
_register_conv_cls("Conv")
_register_conv_cls("ValidConv",  fix=dict(valid_conv=True))
_register_conv_cls("Deconv", fix=dict(deconv=True), default=dict(kernel_size=2, stride=2))
_register_conv_cls("StridedConv",  default=dict(stride=2))
_register_conv_cls("DilatedConv",  fix=dict(dilation=2))
_register_conv_cls("DepthwiseConv", fix=dict(deconv=False, depthwise=True), default=dict(out_channels='auto'))

# BatchNormRelu classes
_register_bnr_conv_cls("Conv", fix=dict(deconv=False))
_register_bnr_conv_cls("Deconv", fix=dict(deconv=True))
_register_bnr_conv_cls("StridedConv",  default=dict(stride=2))
_register_bnr_conv_cls("DilatedConv",  default=dict(dilation=2))
_register_bnr_conv_cls("DepthwiseConv", fix=dict(deconv=False, depthwise=True), default=dict(out_channels='auto'))

del _register_conv_cls
del _register_bnr_conv_cls




class GlobalConv2D(nn.Module):
    """From https://arxiv.org/pdf/1703.02719.pdf
    Main idea: we can have a bigger kernel size computationally acceptable
    if we separate 2D-conv in 2 1D-convs """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        local_conv_type,
        activation=None,
        use_BN=False,
        **kwargs
    ):
        super(GlobalConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert isinstance(kernel_size, (int, list, tuple))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        self.kwargs = kwargs
        self.conv1a = local_conv_type(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(kernel_size[0], 1),
            **kwargs
        )
        self.conv1b = local_conv_type(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(1, kernel_size[1]),
            **kwargs
        )
        self.conv2a = local_conv_type(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, kernel_size[1]),
            **kwargs
        )
        self.conv2b = local_conv_type(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(kernel_size[0], 1),
            **kwargs
        )
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
        out = out1.add(1, out2)
        if self.activation is not None:
            out = self.activation(out)
        if self.batchnorm is not None:
            out = self.batchnorm(out)
        return out
