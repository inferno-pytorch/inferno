import torch
import torch.nn as nn
from ..layers.identity import Identity
from ..layers.convolutional_blocks import ResBlock
from ..layers.convolutional import ConvELU2D, ConvELU3D, Conv2D, Conv3D
from ...utils.math_utils import max_allowed_ds_steps
from ...utils.python_utils import require_dict_kwagrs


__all__ = ['UNetBase', 'ResBlockUNet']
_all = __all__


class UNetBase(nn.Module):

    """ Base class for implementing UNets.
        The depth and dimension of the UNet is flexible.
        The deriving classes must implement
        `conv_op_factory` and can implement
        `upsample_op_factory` and
        `downsample_op_factory`.

    Attributes:
        in_channels (int): Number of input channels.
        dim (int): Spatial dimension of data (must be 2 or 3)
        depth (int): How many down-sampling / up-sampling steps
            shall be performed
        gain (int): Multiplicative increase of channels while going down in the UNet.
            The same factor is used to decrease the number of channels while
            going up in the UNet.
        residual (bool): If residual is true, the output of the down-streams
            are added to the up-stream results.
            Otherwise the results are concatenated.
    """

    def __init__(self, in_channels, dim, depth=3,
                 gain=2, residual=False, upsample_mode=None, p_dropout=None):

        super(UNetBase, self).__init__()

        # early sanity check
        if dim not in [2, 3]:
            raise RuntimeError("UNetBase is only implemented for 2D and 3D")

        # settings related members
        self.in_channels  = int(in_channels)
        self.dim          = int(dim)
        self.depth        = int(depth)
        self.gain         = int(gain)
        self.residual     = bool(residual)
        self.p_dropout = p_dropout

        # members to remember what to store as side output
        self._store_conv_down = []
        self._store_conv_bottom = False
        self._store_conv_up = []

        # number of channels per side output
        self.n_channels_per_output = []

        # members to hold actual nn.Modules / nn.ModuleLists
        self._pre_conv_down_ops  = None
        self._post_conv_down_ops = None
        self._conv_down_ops  = None

        self._pre_conv_up_ops  = None
        self._post_conv_up_ops = None
        self._conv_up_ops = None

        self._upsample_ops = None
        self._downsample_ops = None

        self._pre_conv_bottom_ops  = None
        self._post_conv_bottom_ops = None
        self._conv_bottom_op = None

        # upsample kwargs
        self._upsample_kwargs = self._make_upsample_kwargs(upsample_mode=upsample_mode)

        ########################################
        # default dropout
        ########################################
        if self.p_dropout is not None:
            self.use_dropout = True
            if self.dim == 2 :
                self._channel_dropout_op = self.torch.nn.Dropout2d(p=float(self.p_dropout),
                                                                   inplace=False)
            else:
                self._channel_dropout_op = self.torch.nn.Dropout3d(p=float(self.p_dropout),
                                                                   inplace=False)
        else:
            self.use_dropout = False

        # down-stream convolution blocks
        self._init__downstream()

        # pooling / downsample operators
        self._downsample_ops = nn.ModuleList([
            self.downsample_op_factory(i) for i in range(depth)
        ])

        # upsample operators
        self._upsample_ops = nn.ModuleList([
            self.upsample_op_factory(i) for i in range(depth)
        ])

        # bottom block of the unet
        self._init__bottom()

        # up-stream convolution blocks
        self._init__upstream()

        assert len(self.n_channels_per_output) == self._store_conv_down.count(True) + \
            self._store_conv_up.count(True) + int(self._store_conv_bottom)

    def _init__downstream(self):
        conv_down_ops = []
        self._store_conv_down = []

        current_in_channels = self.in_channels

        for i in range(self.depth):
            out_channels = current_in_channels * self.gain
            op, return_op_res = self.conv_op_factory(in_channels=current_in_channels,
                                                     out_channels=out_channels,
                                                     part='down', index=i)
            conv_down_ops.append(op)
            if return_op_res:
                self.n_channels_per_output.append(out_channels)
                self._store_conv_down.append(True)
            else:
                self._store_conv_down.append(False)

            # increase the number of channels
            current_in_channels *= self.gain

        # store as proper torch ModuleList
        self._conv_down_ops = nn.ModuleList(conv_down_ops)

        return current_in_channels

    def _init__bottom(self):

        conv_up_ops = []

        current_in_channels = self.in_channels* self.gain**self.depth

        factory_res = self.conv_op_factory(in_channels=current_in_channels,
            out_channels=current_in_channels, part='bottom', index=0)
        if isinstance(factory_res, tuple):
            self._conv_bottom_op, self._store_conv_bottom = factory_res
            if self._store_conv_bottom:
                self.n_channels_per_output.append(current_in_channels)
        else:
            self._conv_bottom_op = factory_res
            self._store_conv_bottom = False

    def _init__upstream(self):
        conv_up_ops = []
        current_in_channels = self.in_channels * self.gain**self.depth

        for i in range(self.depth):
            # the number of out channels (are we in the last block?)
            out_channels = current_in_channels // self.gain

            # if not residual we concat which needs twice as many channels
            fac = 1 if self.residual else 2

            op, return_op_res = self.conv_op_factory(in_channels=fac*current_in_channels,
                                                     out_channels=out_channels,
                                                     part='up', index=i)
            conv_up_ops.append(op)
            if return_op_res:
                self.n_channels_per_output.append(out_channels)
                self._store_conv_up.append(True)
            else:
                self._store_conv_up.append(False)

            # decrease the number of input_channels
            current_in_channels //= self.gain

        # store as proper torch ModuleLis
        self._conv_up_ops = nn.ModuleList(conv_up_ops)

        # the last block needs to be stored in any case
        if not self._store_conv_up[-1]:
            self._store_conv_up[-1] = True
            self.n_channels_per_output.append(out_channels)

    def _make_upsample_kwargs(self, upsample_mode):
        """To avoid some waring from pytorch, and some missing implementations
        for the arguments need to be handle carefully in this helper functions

        Args:
            upsample_mode (str): users choice for upsampling  interpolation style.
        """
        if upsample_mode is None:
            if self.dim == 2:
                upsample_mode = 'bilinear'
            elif self.dim == 3:
                # upsample_mode = 'nearest'
                upsample_mode = 'trilinear'

        upsample_kwargs = dict(scale_factor=2, mode=upsample_mode)
        if upsample_mode in ('bilinear', 'trilinear'):
            upsample_kwargs['align_corners'] = False
        return upsample_kwargs

    def _forward_sanity_check(self, input):
        if isinstance(input, tuple):
            raise RuntimeError("tuples of tensors are not supported")
        shape = list(input.size())
        mx = max_allowed_ds_steps(shape=shape[2:2+self.dim], factor=2)
        if mx < self.depth:
            raise RuntimeError("cannot downsample %d times, with shape %s"%
                (self.depth, str(input.size())) )

        if input.size(1) != self.in_channels :
            raise RuntimeError("wrong number of channels: expected %d, got %d"%
                (self.in_channels, input.size(1)))

        if input.dim() != self.dim + 2 :
            raise RuntimeError("wrong number of dim: expected %d, got %d"%
                (self.dim+2, input.dim()))

    def forward(self, input):

        # check if input is suitable
        self._forward_sanity_check(input=input)

        # collect all desired outputs
        side_out = []

        # remember all conv-block results of the downward part
        # of the UNet
        down_res = []

        #################################
        # downwards part
        #################################
        out = input
        for d in range(self.depth):

            out = self._conv_down_ops[d](out)
            #out = self.dropout

            down_res.append(out)

            if self._store_conv_down[d]:
                side_out.append(out)

            out = self._downsample_ops[d](out)

        #################################
        # bottom part
        #################################
        out = self._conv_bottom_op(out)
        if self._store_conv_bottom:
            side_out.append(out)

        #################################
        # upward part
        #################################
        down_res = list(reversed(down_res)) # <- eases indexing
        for d in range(self.depth):

            # upsample
            out = self._upsample_ops[d](out)

            # the result of the downward part
            a = down_res[d]

            # add or concat?
            if self.residual:
                out = a + out
            else:
                out = torch.cat([a,out], 1)

            # the convolutional block
            out = self._conv_up_ops[d](out)

            if self._store_conv_up[d]:
                side_out.append(out)

        # if  len(side_out) == 1 we actually have no side output
        # just the main output
        if len(side_out) == 1:
            return side_out[0]
        else:
            return tuple(side_out)

    def downsample_op_factory(self, index):
        C = nn.MaxPool2d if self.dim == 2 else nn.MaxPool3d
        return C(kernel_size=2, stride=2)

    def upsample_op_factory(self, index):
        return nn.Upsample(**self._upsample_kwargs)

    def pre_conv_op_regularizer_factory(self, in_channels, out_channels, part, index):
            if self.use_dropout and in_channels > 2:
                return self._channel_dropout_op(x)
            else:
                return Identity()

    def post_conv_op_regularizer_factory(self, in_channels, out_channels, part, index):
            return Identity()

    def conv_op_factory(self, in_channels, out_channels, part, index):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def _dropout(self, x):
        if self.use_dropout:
            return self._channel_dropout_op(x)
        else:
            return x


class UNet(UNetBase):
    """
    Default 2d / 3d U-Net implementation following
    TODO cite Ronneberger U-Net papers.
    """
    def __init__(self, in_channels, out_channels, dim,
                 depth=4, initial_features=64, gain=2,
                 final_activation=None, p_dropout=None):
        # convolutional types for inner convolutions and output convolutions
        self.default_conv = ConvELU2D if dim == 2 else ConvELU3D
        last_conv = Conv2D if dim == 2 else Conv3D
        self.out_channels = int(out_channels)

        # init the base class
        super(UNet, self).__init__(in_channels=initial_features, dim=dim,
                                   depth=depth, gain=gain, p_dropout=p_dropout)
        # initial conv layer to go from the number of input channels, which are defined by the data
        # (usually 1 or 3) to the initial number of feature maps
        self._initial_conv = self.default_conv(in_channels, initial_features, 3)

        # get the final output and activation activation
        if isinstance(final_activation, str):
            activation = getattr(nn, final_activation)()
        elif isinstance(final_activation, nn.Module):
            activation = final_activation
        elif final_activation is None:
            activation = None
        else:
            raise NotImplementedError("Activation of type %s is not supported" % type(final_activation))

        if activation is None:
            self._output = last_conv(initial_features, self.out_channels, 1)
        else:
            self._output = nn.Sequential(last_conv(initial_features, self.out_channels, 1),
                                         activation)

    def forward(self, input):
        # TODO implement 2d from 3d input (see neurofire)
        x = self._initial_conv(input)
        x = super(UNet, self).forward(x)
        return self._output(x)

    def conv_op_factory(self, in_channels, out_channels, part, index):

        # is this the first convolutional block?
        first = (part == 'down' and index == 0)

        # if this is the first conv block, we just need
        # a single convolution, because we have the `_initial_conv` already
        if first:
            conv = self.default_conv(in_channels, out_channels, 3)
        else:
            conv = nn.Sequential(self.default_conv(in_channels, out_channels, 3),
                                 self.default_conv(out_channels, out_channels, 3))
        return conv, False


# TODO rename ResidualBlockUNet
# TODO test
# or ResidualUNet
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
        self.unet_kwargs = require_dict_kwagrs(unet_kwargs, "unet_kwargs must be a dict or None")
        self.res_block_kwargs = require_dict_kwagrs(res_block_kwargs,
                                                    "res_block_kwargs must be a dict or None")
        self.activated = activated
        if isinstance(side_out_parts, str):
            self.side_out_parts = set([side_out_parts])
        elif isinstance(side_out_parts, (tuple,list)):
            self.side_out_parts = set(side_out_parts)
        else:
            self.side_out_parts = set()

        super(ResBlockUNet, self).__init__(in_channels=in_channels,
                                           dim=dim,
                                           out_channels=out_channels,
                                           **self.unet_kwargs)

    def conv_op_factory(self, in_channels, out_channels, part, index):

        # is this the very last convolutional block?
        very_last = (part == 'up' and index + 1 == self.depth)

        # should the residual block be activated?
        activated = not very_last or self.activated

        # should the output be part of the overall
        # return-list in the forward pass of the UNet
        use_as_output = part in self.side_out_parts

        # residual block used within the UNet
        # TODO rename to ResidualBlock
        return ResBlock(in_channels=in_channels, out_channels=out_channels,
                             dim=self.dim, activated=activated,
                             **self.res_block_kwargs), use_as_output
