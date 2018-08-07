from inferno.extensions.layers.convolutional import ConvELU2D
import torch 
import torch.nn as nn


from .building_blocks import ResBlock





# lil helper
def ensure_dict_kwagrs(kwargs, msg=None):
    if kwargs is None:
        return dict()
    elif isinstance(kwargs, dict):
        return dict()
    else:
        if msg is None:
            raise RuntimeError("value passed as keyword argument dict is neither none nor a  dict")
        else:
            raise RuntimeError("%s"%str(msg))




def max_allowed_ds_steps(shape, factor):
    def max_allowed_ds_steps_impl(size, factor):

        current_size = float(size)
        allowed_steps = 0 
        while(True):

            new_size = current_size / float(factor)
            if(new_size >=1 and new_size.is_integer()):

                current_size = new_size
                allowed_steps += 1
            else:
                break
        return allowed_steps

    min_steps = float('inf')

    for s in shape:
        min_steps = int(min(min_steps, max_allowed_ds_steps_impl(s, factor)))

    return min_steps

class UNetBase(nn.Module):
    def __init__(self, in_channels, out_channels, dim, depth=3, gain=2, residual=False, upsample_mode=None):

        super(UNetBase, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.dim = int(dim)    
        self.depth = int(depth)
        self.gain = int(gain)
        self.residual = bool(residual)
        self.upsample_mode = upsample_mode

        if upsample_mode is None:
            if dim == 2:
                self.upsample_mode = 'bilinear'
            elif dim == 3:
                self.upsample_mode = 'nearest'
            else:
                raise RuntimeError("unet is only implemented for 2d and 3d not for %d-d"%self.dim)


        self.upsample_kwargs = dict(scale_factor=2, mode=self.upsample_mode)
        if self.upsample_mode == 'bilinear':
            self.upsample_kwargs['align_corners'] = False


        in_c  = in_channels

        # convolution block downwards
        self.store_conv_down = []
        self.store_conv_bottom = False
        self.store_conv_up = []

        conv_down_ops = []
        for i in range(depth):
            out_c = in_c * self.gain
            conv_and_store = self.conv_op_factory(in_channels=in_c, out_channels=out_c, part='down',index=i)
            if isinstance(conv_and_store, tuple):
                conv_down_ops.append(conv_and_store[0])
                self.store_conv_down.append(conv_and_store[1])
            else:
                conv_down_ops.append(conv_and_store)
                self.store_conv_down.append(False)
            in_c *= gain
         




        self.conv_down_ops = nn.ModuleList(conv_down_ops)

        # pooling / down-sample operators
        self.downsample_ops = nn.ModuleList([
            self.downsample_op_factory(i) for i in range(depth)
        ])

        
        # upsample operators 
        self.upsample_ops = nn.ModuleList([
            self.upsample_op_factory(i) for i in range(depth)
        ])

        # bottom block
        conv_bottom_op_and_store = self.conv_op_factory(in_channels=in_c, out_channels=in_c, part='bottom', index=0)
        if isinstance(conv_bottom_op_and_store, tuple):

            self.conv_bottom_op = conv_bottom_op_and_store[0]
            self.store_conv_bottom = conv_bottom_op_and_store[1]
        else:
            self.conv_bottom_op = conv_bottom_op_and_store
            self.store_conv_bottom = False
    

        # convolution block upwards
        conv_up_ops = []
        for i in range(depth):

            # if not residual we concat which needs twice as many channels
            fac = 1 if self.residual else 2

            # are we in the last block?
            out_c = self.out_channels if (i+1 == depth) else in_c//gain

            conv_and_store = self.conv_op_factory(in_channels=fac*in_c, out_channels=out_c, part='up', index=i)
            if isinstance(conv_and_store, tuple):
                conv_up_ops.append(conv_and_store[0])
                self.store_conv_up.append(conv_and_store[1])
            else:
                conv_up_ops.append(conv_and_store)
                self.store_conv_up.append(False)

            in_c //= gain
        

        # the last block needs to be stores in any case
        self.store_conv_up[-1] = True

        self.conv_up_ops = nn.ModuleList(conv_up_ops)

    
    def downsample_op_factory(self, i):
        if self.dim == 2:
            return nn.MaxPool2d(kernel_size=2, stride=2) 
        else:
            return nn.MaxPool3d(kernel_size=2, stride=2) 

    def upsample_op_factory(self, i):
        return nn.Upsample(**self.upsample_kwargs)



    def conv_op_factory(self, in_channels, out_channels, part, index):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")



    def _forward_sanity_check(self, input):
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

        # collect all outputs
        side_out = []

        assert input.size(1) == self.in_channels
        down_res = []

        # downwards
        out = input
        for d in range(self.depth):


            out = self.conv_down_ops[d](out)
            #res = self.dropout(res)
            down_res.append(out)

            if self.store_conv_down[d]:
                side_out.append(out)

            out = self.downsample_ops[d](out)



        
        out = self.conv_bottom_op(out)
        if self.store_conv_bottom:
            side_out.append(out)

        # reverse list for easier indexing
        down_res = list(reversed(down_res))

        # upwards
        for d in range(self.depth):

            # upsample
            out = self.upsample_ops[d](out)

        
            a = down_res[d]

            if self.residual:
                out = a + out
            else:
                out = torch.cat([a,out], 1)

            # conv
            out = self.conv_up_ops[d](out)

            if self.store_conv_up[d]:
                side_out.append(out)
    
        assert out.size(1) == self.out_channels
        
        # if  len(side_out) == 1 we actually have no side output
        # just the main output 
        if len(side_out) == 1:
            return side_out[0]
        else:
            return tuple(side_out)




class ResBlockUNet(UNetBase):
    def __init__(self, in_channels, dim, out_channels, unet_kwargs=None, 
                 res_block_kwargs=None, activated=True,
                 side_out_parts=None
        ):


        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_block_kwargs = res_block_kwargs
        
        self.unet_kwargs      = ensure_dict_kwagrs(unet_kwargs,      "unet_kwargs must be a dict or None")
        self.res_block_kwargs = ensure_dict_kwagrs(res_block_kwargs, "res_block_kwargs must be a dict or None")
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
        very_last = part == 'up' and index + 1 == self.depth
        activated = not very_last or self.activated

        use_as_output = part in self.side_out_parts

        return ResBlock(in_channels=in_channels, out_channels=out_channels, 
                             dim=self.dim, activated=activated,
                             **self.res_block_kwargs), use_as_output