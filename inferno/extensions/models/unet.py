from collections import OrderedDict
import torch
import torch.nn as nn
from ..layers.identity import Identity
from ..layers.activations import get_activation
from ..layers.convolutional import *#ConvELU2D, ConvELU3D, Conv2D, Conv3D
from ..layers.sampling import Upsample as InfernoUpsample
from ...utils.math_utils import max_allowed_ds_steps


__all__ = ['UNetBase', 'UNet', 'ResBlockUNet']
_all = __all__


class UNetBase(nn.Module):

    """ Base class for implementing UNets.
        The depth and dimension of the UNet is flexible.
        The deriving classes must implement
        `conv_op_factory` and can implement
        `upsample_op_factory`,
        `downsample_op_factory` and
        .

    Attributes:
        dim (int): Spatial dimension of data (must be 1, 2 or 3).
        in_channels (int): Number of input channels.
        initial_features (int): Number of desired features after initial conv block
        out_channels (int): Number of output channels. Set to None by default,
            which sets the number of output channels to get_num_channels(0) x initial (default: None).
        depth (int): How many down-sampling / up-sampling steps
            shall be performed (default: 3).
        gain (int): Multiplicative increase of channels while going down in the UNet.
            The same factor is used to decrease the number of channels while
            going up in the UNet (default: 2).
            If UNetBase.get_num_channels is overwritten, this parameter becomes meaningless.
        residual (bool): If residual is true, the output of the down-streams
            are added to the up-stream results.
            Otherwise the results are concatenated (default: False).
    """

    def __init__(self, dim, in_channels, initial_features, out_channels=None, depth=3,
                 gain=2, residual=False, upsample_mode=None):

        super(UNetBase, self).__init__()

        # early sanity check
        if dim not in [1, 2, 3]:
            raise RuntimeError("UNetBase is only implemented for 1D, 2D and 3D")

        # settings related members
        self.in_channels        = int(in_channels)
        self.initial_features   = int(initial_features)
        self.dim                = int(dim)

        if out_channels is None:
            self.out_channels = self.get_num_channels(1)
        else:
            self.out_channels = int(out_channels)

        self.depth        = int(depth)
        self.gain         = int(gain)
        self.residual     = bool(residual)
        

        # members to remember what to store as side output
        self._side_out_num_channels = OrderedDict()
        # and number of channels per side output
        self.n_channels_per_output = None 


        # members to hold actual nn.Modules / nn.ModuleLists
        self._conv_start_op = None
        self._conv_down_ops  = None
        self._downsample_ops = None
        self._conv_bottom_op = None
        self._upsample_ops = None
        self._conv_up_ops = None
        self._conv_end_op = None

        # upsample kwargs
        self._upsample_kwargs = self._make_upsample_kwargs(upsample_mode=upsample_mode)



        # initialize all parts of the unet
        # (do not change order since we use ordered dict
        # to remember number of out channels of side outs)
        # - convs
        self._init__start()
        self._init__downstream()
        self._init__bottom()
        self._init__upstream()
        self._init__end()
        # - pool/upsample downsample
        self._init__downsample()
        self._init__upsample()

        # side out related 
        n_outputs = len(self._side_out_num_channels)
        self.out_channels_side_out = tuple(self._side_out_num_channels.values())

    def get_num_channels(self, depth):
        #assert depth > 0
        return self.initial_features * self.gain**depth



    def _init__downstream(self):
        conv_down_ops = []
        self._store_conv_down = []

        current_in_channels = self.initial_features

        for i in range(self.depth):
            out_channels = self.get_num_channels(i + 1)
            op, return_op_res = self.conv_op_factory(in_channels=current_in_channels,
                                                     out_channels=out_channels,
                                                     part='down', index=i)
            conv_down_ops.append(op)
            if return_op_res:
                self._side_out_num_channels[('down', i)] = out_channels
    
            # increase the number of channels
            current_in_channels = out_channels

        # store as proper torch ModuleList
        self._conv_down_ops = nn.ModuleList(conv_down_ops)

        return current_in_channels

    def _init__downsample(self):
        # pooling / downsample operators
        self._downsample_ops = nn.ModuleList([
            self.downsample_op_factory(i) for i in range(self.depth)
        ])

    def _init__upsample(self):
        # upsample operators
        # we flip the index that is given as argument to index consistently in up and
        # downstream sampling factories
        self._upsample_ops = nn.ModuleList([
            self.upsample_op_factory(self._inv_index(i)) for i in range(self.depth)
        ])

    def _init__bottom(self):

        current_in_channels = self.get_num_channels(self.depth)

        op, return_op_res = self.conv_op_factory(in_channels=current_in_channels,
            out_channels=current_in_channels, part='bottom', index=None)
        self._conv_bottom_op = op
        if return_op_res:
            self._side_out_num_channels['bottom'] = current_in_channels


    def _init__upstream(self):
        conv_up_ops = []
        current_in_channels = self.get_num_channels(self.depth)

        for i in range(self.depth):

            # the number of out channels 
            if i + 1 < self.depth:
                out_channels = self.get_num_channels(self._inv_index(i))
            else:
                out_channels = self.initial_features

            # if not residual we concat which needs twice as many channels
            fac = 1 if self.residual else 2

            # we flip the index that is given as argument to index consistently in up and
            # downstream conv factories
            op, return_op_res = self.conv_op_factory(in_channels=fac*current_in_channels,
                                                     out_channels=out_channels,
                                                     part='up', index=self._inv_index(i))
            conv_up_ops.append(op)
            if return_op_res:
                self._side_out_num_channels[('up', index)]

            # decrease the number of input_channels
            current_in_channels = out_channels

        # store as proper torch ModuleLis
        self._conv_up_ops = nn.ModuleList(conv_up_ops)

    def _init__start(self):
        conv, return_op_res = self.conv_op_factory(in_channels=self.in_channels,
                                                     out_channels=self.initial_features,
                                                     part='start', index=None)
        if return_op_res:
            self._side_out_num_channels['start'] = self.initial_features
     
        # since this is the very last layer of the unet
        # we ALWAYS return the result of this op
        # and ignore return_op_res
        self._side_out_num_channels['end'] = self.out_channels

        self._start_block = conv 
    
    def _init__end(self):
        conv, return_op_res = self.conv_op_factory(in_channels=self.get_num_channels(0),
                                                   out_channels=self.out_channels,
                                                     part='end', index=None)
        # since this is the very last layer of the unet
        # we ALWAYS return the result of this op
        # and ignore return_op_res
        self._side_out_num_channels['end'] = self.out_channels

        self._end_block = conv  

    def _make_upsample_kwargs(self, upsample_mode):
        """To avoid some waring from pytorch, and some missing implementations
        for the arguments need to be handle carefully in this helper functions

        Args:
            upsample_mode (str): users choice for upsampling  interpolation style.
        """
        if upsample_mode is None:
            if self.dim == 1:
                upsample_mode = 'linear'
            elif self.dim == 2:
                upsample_mode = 'bilinear'
            elif self.dim == 3:
                # upsample_mode = 'nearest'
                upsample_mode = 'trilinear'

        upsample_kwargs = dict(scale_factor=2, mode=upsample_mode)
        if upsample_mode in ('linear','bilinear', 'trilinear'):
            upsample_kwargs['align_corners'] = False
        return upsample_kwargs

    def _forward_sanity_check(self, input):
        if isinstance(input, tuple):
            raise RuntimeError("tuples of tensors are not supported")
        shape = input.shape

        if shape[1] != self.in_channels:
            raise RuntimeError("wrong number of channels: expected %d, got %d"%
                (self.in_channels, input.size(1)))

        if input.dim() != self.dim + 2:
            raise RuntimeError("wrong number of dim: expected %d, got %d"%
                (self.dim+2, input.dim()))
        self._check_scaling(input)

    # override if model has different scaling
    def _check_scaling(self, input):
        shape = input.shape
        mx = max_allowed_ds_steps(shape=shape[2:2+self.dim], factor=2)
        if mx < self.depth:
            raise RuntimeError("cannot downsample %d times, with shape %s"%
                (self.depth, str(input.size())) )

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
        out = self._start_block(out)
        if 'start' in  self._side_out_num_channels:
            side_out.append(out)
        for d in range(self.depth):

            out = self._conv_down_ops[d](out)
            #out = self.dropout

            down_res.append(out)

            if ('down',d) in  self._side_out_num_channels:
                side_out.append(out)

            out = self._downsample_ops[d](out)

        #################################
        # bottom part
        #################################
        out = self._conv_bottom_op(out)
        if 'bottom' in  self._side_out_num_channels:
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
                out = torch.cat([a, out], 1)

            # the convolutional block
            out = self._conv_up_ops[d](out)

            if ('up', self._inv_index(d)) in  self._side_out_num_channels:
                side_out.append(out)


        out  = self._end_block(out)
        #always return last block ``if 'end' in  self._side_out_num_channels:``
        side_out.append(out)

        # if we only have the last layer as output
        # we return a single tensor, otherwise a tuple of
        # tensor
        if len(side_out) == 1:
            return side_out[0]
        else:
            return tuple(side_out)

    def downsample_op_factory(self, index):
        if self.dim == 1:
            return nn.MaxPool1d(kernel_size=2, stride=2)
        elif self.dim == 2:
            return nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.dim == 3:
            return nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            # should be nonreachable
            assert False

    def upsample_op_factory(self, index):
        return InfernoUpsample(**self._upsample_kwargs)
        #return nn.Upsample(**self._upsample_kwargs)

    def conv_op_factory(self, in_channels, out_channels, part, index):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def _inv_index(self, index):
        # we flip the index that is given as argument to index consistently in up and
        # downstream conv factories
        return self.depth - 1 - index





# TODO implement function to load a pretrained unet
class UNet(UNetBase):
    """
    Default 2d / 3d U-Net implementation following:
    https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_channels, out_channels, dim,
                 depth=4, initial_features=64, gain=2,
                 final_activation=None):
        # convolutional types for inner convolutions and output convolutions
        self.dim = dim
        self.default_conv = ConvELU


        # init the base class
        super(UNet, self).__init__(in_channels=in_channels, initial_features=initial_features, dim=dim,
                                   out_channels=out_channels, depth=depth, gain=gain)


        # get the final output and activation activation
        activation = get_activation(final_activation)
 
        # override the unet base attributes for out_channels
        # if activation is None:
        #     self._output = self.default_conv(dim, initial_features*gain, self.out_channels, 1)
        # else:
        #     self._output = nn.Sequential(
        #             self.default_conv(dim, initial_features*gain, self.out_channels, 1),
        #             activation)

    # def forward(self, input):
    #     # TODO implement 2d from 3d input (see neurofire)
    #     x = self._initial_conv(input)
    #     x = super(UNet, self).forward(x)
    #     return self._output(x)

    def conv_op_factory(self, in_channels, out_channels, part, index):

        # is this the first convolutional block?
        first = (part == 'down' and index == 0)

        # initial block or first block just have one convolution
        if  part == 'start' or (part == 'down' and index == 0):
            conv = self.default_conv(self.dim, in_channels, out_channels, 3)
        elif part == 'end':
            conv = self.default_conv(self.dim, self.initial_features, self.out_channels, 1)
        else:
            conv = nn.Sequential(self.default_conv(self.dim, in_channels, out_channels, 3),
                                 self.default_conv(self.dim, out_channels, out_channels, 3))
        return conv, False
