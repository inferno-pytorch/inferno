import torch
import torch.nn as nn
from ..layers.convolutional import ConvActivation
from .unet import UNetBase
from ...utils.python_utils import require_dict_kwargs

__all__ = ['ResBlockUNet']
_all = __all__



# We only use this for the u-net implementation here
# in favor of less code duplication it might be a
# good ideat to replace this with 'ResidualBlock' from layers.convolutional_blocks
class _ResBlockBase(nn.Module):
    def __init__(self, in_channels, out_channels, dim,
                 size=2, force_skip_op=False, activated=True):
        super(_ResBlockBase, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.size = int(size)
        self.activated = bool(activated)
        self.force_skip_op = bool(force_skip_op)
        self.dim = int(dim)

        if self.in_channels != self.out_channels or self.force_skip_op:
            self.activated_skip_op = self.activated_skip_op_factory(in_channels=self.in_channels,
                                                                    out_channels=self.out_channels)

        conv_ops = []
        activation_ops = []
        for i in range(self.size):

            # the convolutions
            if i == 0:
                op = self.nonactivated_conv_op_factory(in_channels=self.out_channels,
                                                       out_channels=self.out_channels, index=i)
            else:
                op = self.nonactivated_conv_op_factory(in_channels=self.out_channels,
                                                       out_channels=self.out_channels, index=i)
            conv_ops.append(op)

            # the activations
            if i < self.size or self.activated:
                activation_ops.append(self.activation_op_factory(index=i))

        self.conv_ops = nn.ModuleList(conv_ops)
        self.activation_ops = nn.ModuleList(activation_ops)

    def activated_skip_op_factory(self, in_channels, out_channels):
        raise NotImplementedError("activated_skip_op_factory need to be implemented by deriving class")

    def nonactivated_conv_op_factory(self, in_channels, out_channels, index):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def activation_op_factory(self, index):
        return nn.ReLU()

    def forward(self, input):

        if input.size(1) != self.in_channels:
            raise RuntimeError("wrong number of channels: expected %d, got %d"%
                (self.in_channels, input.size(1)))

        if input.dim() != self.dim + 2:
            raise RuntimeError("wrong number of dim: expected %d, got %d"%
                (self.dim+2, input.dim()))

        if self.in_channels != self.out_channels or self.force_skip_op:
            skip_res = self.activated_skip_op(input)
        else:
            skip_res = input

        assert skip_res.size(1) == self.out_channels

        res = skip_res
        for i in  range(self.size):
            res = self.conv_ops[i](res)
            assert res.size(1)  == self.out_channels
            if i + 1 < self.size:
                res = self.activation_ops[i](res)

        non_activated = skip_res + res
        if self.activated:
            return self.activation_ops[-1](non_activated)
        else:
            return non_activated


class _ResBlock(_ResBlockBase):
    def __init__(self, in_channels, out_channels, dim, size=2, activated=True,
                 activation='ReLU', batchnorm=True, force_skip_op=False, conv_kwargs=None):

        # trick to store  nn-module before call of super
        # => we put it in a list
        if isinstance(activation, str):
            self.activation_op = [getattr(torch.nn, activation)()]
        elif isinstance(activation, nn.Module):
            self.activation_op = [activation]
        else:
            raise RuntimeError("activation must be a striong or a torch.nn.Module")

        # keywords for conv
        if conv_kwargs is None:
            conv_kwargs = dict(
                 kernel_size=3, dim=dim, activation=None,
                 stride=1, dilation=1, groups=None, depthwise=False, bias=True,
                 deconv=False, initialization=None
            )
        elif isinstance(conv_kwargs, dict):
            conv_kwargs['activation'] = None
        else:
            raise RuntimeError("conv_kwargs must be either None or a dict")
        self.conv_kwargs = conv_kwargs

        self.dim = dim
        self.batchnorm = batchnorm

        self.conv_1x1_kwargs = dict(kernel_size=1, dim=dim, activation=None,
                                    stride=1, dilation=1, groups=None,
                                    depthwise=False, bias=True, deconv=False,
                                    initialization=None)

        super(_ResBlock, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        dim=dim, size=size,
                                        force_skip_op=force_skip_op,
                                        activated=activated)

    def activated_skip_op_factory(self, in_channels, out_channels):
        conv_op = ConvActivation(in_channels=in_channels,
                                 out_channels=out_channels, **self.conv_1x1_kwargs)
        if self.batchnorm:
            batchnorm_op = self.batchnorm_op_factory(in_channels=out_channels)
            return torch.nn.Sequential(conv_op, batchnorm_op, self.activation_op[0])
        else:
            return torch.nn.Sequential(conv_op, self.activation_op[0])

    def nonactivated_conv_op_factory(self, in_channels, out_channels, index):
        conv_op = ConvActivation(in_channels=in_channels,
                                 out_channels=out_channels, **self.conv_kwargs)
        if self.batchnorm:
            batchnorm_op = self.batchnorm_op_factory(in_channels=out_channels)
            return torch.nn.Sequential(conv_op, batchnorm_op)
        else:
            return conv_op

    def activation_op_factory(self, index):
        return self.activation_op[0]

    def batchnorm_op_factory(self, in_channels):
        bn_cls_name = 'BatchNorm{}d'.format(int(self.dim))
        bn_op_cls = getattr(torch.nn, bn_cls_name)
        return bn_op_cls(in_channels)


# TODO not sure how to handle out-channels properly.
# For now, we just force the corrcect number in the last decoder layer
class ResBlockUNet(UNetBase):
    """TODO.

        ACCC

    Attributes:
        activated (TYPE): Description
        dim (TYPE): Description
        res_block_kwargs (TYPE): Description
        side_out_parts (TYPE): Description
        unet_kwargs (TYPE): Description
    """
    def __init__(self, in_channels, dim, out_channels, unet_kwargs=None,
                 res_block_kwargs=None, activated=True,
                 side_out_parts=None):

        self.dim = dim
        self.unet_kwargs = require_dict_kwargs(unet_kwargs, "unet_kwargs must be a dict or None")
        self.res_block_kwargs = require_dict_kwargs(res_block_kwargs,
                                                    "res_block_kwargs must be a dict or None")
        self.activated = activated
        if isinstance(side_out_parts, str):
            self.side_out_parts = set([side_out_parts])
        elif isinstance(side_out_parts, (tuple,list)):
            self.side_out_parts = set(side_out_parts)
        else:
            self.side_out_parts = set()

        super(ResBlockUNet, self).__init__(in_channels=in_channels,
                                           out_channels=out_channels,
                                           dim=dim,
                                           **self.unet_kwargs)

    def conv_op_factory(self, in_channels, out_channels, part, index):

        # is this the very last convolutional block?
        very_last = (part == 'up' and index == 0)

        # should the residual block be activated?
        activated = not very_last or self.activated

        # should the output be part of the overall
        # return-list in the forward pass of the UNet
        use_as_output = part in self.side_out_parts

        # residual block used within the UNet
        return _ResBlock(in_channels=in_channels, out_channels=out_channels,
                         dim=self.dim, activated=activated,
                         **self.res_block_kwargs), use_as_output
