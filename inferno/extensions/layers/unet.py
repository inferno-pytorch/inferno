from inferno.extensions.layers.convolutional import ConvELU2D
import torch 
import torch.nn as nn
from functools import partial

from .building_blocks import ResBlock

from ...utils.python_utils import require_dict_kwagrs
from ...utils.math_utils import max_allowed_ds_steps


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
        in_channels (int): Description
        out_channels (int): Description
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

    def __init__(self, in_channels, out_channels, dim, depth=3, gain=2, residual=False, upsample_mode=None):

        super(UNetBase, self).__init__()

        # early sanity check
        if dim not in [2, 3]:
            raise RuntimeError("UNetBase is only implemented for 2D and 3D")

        # settings related members
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.dim          = int(dim)    
        self.depth        = int(depth)
        self.gain         = int(gain)
        self.residual     = bool(residual)

        # members to remember what to store as side output
        self.__store_conv_down = []
        self.__store_conv_bottom = False
        self.__store_conv_up = []

        # number of channels per side output
        self.n_channels_per_output = []
        

        # members to hold actual nn.Modules / nn.ModuleLists
        self.__conv_down_ops = None
        self.__conv_up_ops = None
        self.__upsample_ops = None
        self.__downsample_ops = None
        self.__conv_bottom_op = None

        # upsample kwargs 
        self.__upsample_kwargs = self._make_upsample_kwargs(upsample_mode=upsample_mode)


        # the conv. factory functions can return either a module
        # or a module and a bool which indicated if the output
        # of conv block shall be used as side output.
        # If only a module is returned, this is an implicit `False`.
        # This function eases the adding of factory results to lists
        def add_conv_op(res, conv_list, store_res_list, out_channels):
            if isinstance(res, tuple):
                conv_list.append(res[0])
                store_res_list.append(res[1])
                if res[1]:
                    self.n_channels_per_output.append(out_channels)
            else:
                conv_list.append(res)
                store_res_list.append(False)
            return conv_list, store_res_list

        
        ########################################
        # down-stream convolution blocks
        ########################################
        conv_down_ops = [] 
        add_conv_down_op = partial(add_conv_op, conv_list=conv_down_ops, 
                                   store_res_list=self.__store_conv_down)

        current_in_channels = in_channels
        
        for i in range(depth):
            factory_res = self.conv_op_factory(in_channels=current_in_channels, 
                                              out_channels=current_in_channels * self.gain,  
                                              part='down',index=i)
            add_conv_down_op(factory_res, out_channels=current_in_channels * self.gain)

            # increase the number of channels
            current_in_channels *= gain

        # store as proper torch ModuleList
        self.__conv_down_ops = nn.ModuleList(conv_down_ops)


        ########################################
        # pooling / downsample operators
        ########################################
        self.__downsample_ops = nn.ModuleList([
            self.downsample_op_factory(i) for i in range(depth)
        ])
        
        ########################################
        # upsample operators
        ########################################
        self.__upsample_ops = nn.ModuleList([
            self.upsample_op_factory(i) for i in range(depth)
        ])

        ########################################
        # bottom block of the unet
        ########################################       
        conv_up_ops = []



        factory_res = self.conv_op_factory(in_channels=current_in_channels, 
            out_channels=current_in_channels, part='bottom', index=0)
        if isinstance(factory_res, tuple):
            self.__conv_bottom_op, self.__store_conv_bottom = factory_res
            if self.__store_conv_bottom:
                self.n_channels_per_output.append(current_in_channels)
        else:
            self.__conv_bottom_op = factory_res
            self.__store_conv_bottom = False
    

        ########################################
        # up-stream convolution blocks
        ########################################
        conv_up_ops = []
        add_conv_up_op = partial(add_conv_op, conv_list=conv_up_ops, 
                           store_res_list=self.__store_conv_up)
        for i in range(depth):

            

            # the number of out channels (are we in the last block?)
            out_c = self.out_channels if (i+1 == depth) else current_in_channels//gain

            # if not residual we concat which needs twice as many channels
            fac = 1 if self.residual else 2

            factory_res = self.conv_op_factory(in_channels=fac*current_in_channels, 
                                              out_channels=out_c, part='up', index=i)
            add_conv_up_op(factory_res, out_channels=out_c)

            # decrease the number of input_channels
            current_in_channels //= gain

        # store as proper torch ModuleLis
        self.__conv_up_ops = nn.ModuleList(conv_up_ops)

        # the last block needs to be stored in any case
        if not self.__store_conv_up[-1]:
            self.__store_conv_up[-1] = True
            self.n_channels_per_output.append(self.out_channels)

       

        assert len(self.n_channels_per_output) == self.__store_conv_down.count(True) + \
            self.__store_conv_up.count(True)   + int(self.__store_conv_bottom)
        

    
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
                upsample_mode = 'nearest'

        upsample_kwargs = dict(scale_factor=2, mode=upsample_mode)
        if upsample_mode == 'bilinear':
            upsample_kwargs['align_corners'] = False
        return upsample_kwargs

    def __forward_sanity_check(self, input):
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
        self.__forward_sanity_check(input=input)

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


            out = self.__conv_down_ops[d](out)
            down_res.append(out)

            if self.__store_conv_down[d]:
                side_out.append(out)

            out = self.__downsample_ops[d](out)

        
        #################################
        # bottom part
        #################################
        out = self.__conv_bottom_op(out)
        if self.__store_conv_bottom:
            side_out.append(out)


        #################################
        # upward part
        #################################
        down_res = list(reversed(down_res)) # <- eases indexing
        for d in range(self.depth):

            # upsample
            out = self.__upsample_ops[d](out)

            # the result of the downward part
            a = down_res[d]

            # add or concat?
            if self.residual:
                out = a + out
            else:
                out = torch.cat([a,out], 1)

            # the convolutional block
            out = self.__conv_up_ops[d](out)

            if self.__store_conv_up[d]:
                side_out.append(out)
        
        # debug postcondition
        assert out.size(1) == self.out_channels, "internal error"
        
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
        return nn.Upsample(**self.__upsample_kwargs)

    def conv_op_factory(self, in_channels, out_channels, part, index):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

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
                 side_out_parts=None
        ):

        self.dim = dim
        self.unet_kwargs      = require_dict_kwagrs(unet_kwargs,      "unet_kwargs must be a dict or None")
        self.res_block_kwargs = require_dict_kwagrs(res_block_kwargs, "res_block_kwargs must be a dict or None")
        self.activated = activated
        if isinstance(side_out_parts, str):
            self.side_out_parts = set([side_out_parts])
        elif isinstance(side_out_parts, (tuple,list)):
            self.side_out_parts = set(side_out_parts)
        else:
            self.side_out_parts = set()

        super(ResBlockUNet, self).__init__(
            in_channels=in_channels, 
            dim=dim,
            out_channels=out_channels, 
            **self.unet_kwargs
        )



    def conv_op_factory(self, in_channels, out_channels, part, index):

        # is this the very last convolutional block?
        very_last = (part == 'up' and index + 1 == self.depth)


        # should the residual block be activated?
        activated = not very_last or self.activated

        # should the output be part of the overall 
        # return-list in the forward pass of the UNet
        use_as_output = part in self.side_out_parts

        # residual block used within the UNet
        return ResBlock(in_channels=in_channels, out_channels=out_channels, 
                             dim=self.dim, activated=activated,
                             **self.res_block_kwargs), use_as_output