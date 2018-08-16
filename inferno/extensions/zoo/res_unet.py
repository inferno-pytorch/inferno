import torch 
import torch.nn as nn

from .building_blocks import ResBlock
from .unet_base import UNetBase
from ...utils.python_utils import require_dict_kwagrs


__all__ = ['ResBlockUNet']
_all = __all__

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