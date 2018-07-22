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


        conv_in_channels  = in_channels
        # convolution block downwards
        conv_down_ops = []
        for i in range(depth):
            conv = self.conv_down_op_factory(in_channels=conv_in_channels, out_channels=conv_in_channels*self.gain,
                i=i)
            conv_in_channels *= gain

            conv_down_ops.append(conv)

        self.conv_down_ops = nn.ModuleList(conv_down_ops)

        # pooling  downsample
        self.downsample_ops = nn.ModuleList([
            self.downsample_op_factory(i) for i in range(depth)
        ])

        

        # upsample 
        self.upsample_ops = nn.ModuleList([
            self.upsample_op_factory(i) for i in range(depth)
        ])

        # bottom block
        #self.conv_bottom_op = self.conv_bottom_op_factory(conv_in_channels,int(conv_in_channels*gain))
        self.conv_bottom_op = self.conv_bottom_op_factory(conv_in_channels, conv_in_channels)
        #conv_in_channels *= gain

        # convolution block upwards
        conv_up_ops = []
        for i in range(depth):
            # the last block needs special handling

            # if not residual we concat
            fac = 1 
            if not self.residual :
                fac = 2

            # are we in the last block?
            if i + 1 == depth:
                conv = self.conv_up_op_factory( in_channels=fac*conv_in_channels, out_channels=self.out_channels, i=i)
                conv_in_channels //= gain
                conv_up_ops.append(conv)
            else:
                conv = self.conv_up_op_factory(in_channels=fac*conv_in_channels, out_channels=conv_in_channels//gain, i=i)
                conv_in_channels //= gain
                conv_up_ops.append(conv)

        self.conv_up_ops = nn.ModuleList(conv_up_ops)

    
    def downsample_op_factory(self, i):
        if self.dim == 2:
            return nn.MaxPool2d(kernel_size=2, stride=2) 
        else:
            return nn.MaxPool3d(kernel_size=2, stride=2) 

    def upsample_op_factory(self, i):
        return nn.Upsample(**self.upsample_kwargs)

    def conv_op_factory(self, in_channels, out_channels, last):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def conv_down_op_factory(self, in_channels, out_channels, i):
        return self.conv_op_factory(in_channels=in_channels, out_channels=out_channels, last=False)

    def conv_up_op_factory(self, in_channels, out_channels, i):
        is_very_last_conv = bool(i+1 == self.depth)
        return self.conv_op_factory(in_channels=in_channels, out_channels=out_channels, last=is_very_last_conv)

    def conv_bottom_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels, out_channels=out_channels, last=False)

    def forward(self, input):

        # collect all outputs
        side_out = []

        assert input.size(1) == self.in_channels
        down_res = []

        # downwards
        for d in range(self.depth):


            res = self.conv_down_ops[d](input)
            #res = self.dropout(res)
            down_res.append(res)
            input = self.downsample_ops[d](res)
        
        input = self.conv_bottom_op(input)

        down_res = list(reversed(down_res))

        # the first 
        side_out.append(input)

        # upwards
        for d in range(self.depth):

            # upsample
            input = self.upsample_ops[d](input)

        
            a = down_res[d]

            if self.residual:
                input = a + input
            else:
                input = torch.cat([a,input], 1)

            # conv
            input = self.conv_up_ops[d](input)
            #if d + 1 != self.depth:
            #    input = self.dropout(input)

        assert input.size(1) == self.out_channels
        return input 



class ResBlockUNet(UNetBase):
    def __init__(self, in_channels, dim, out_channels, unet_kwargs=None, 
                 res_block_kwargs=None, activated=True
        ):


        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.res_block_kwargs = res_block_kwargs
        
        self.unet_kwargs      = ensure_dict_kwagrs(unet_kwargs,      "unet_kwargs must be a dict or None")
        self.res_block_kwargs = ensure_dict_kwagrs(res_block_kwargs, "res_block_kwargs must be a dict or None")
        self.activated = activated
        super(ResBlockUNet, self).__init__(
            in_channels=in_channels, 
            dim=dim,
            out_channels=out_channels, 
            **self.unet_kwargs
        )

        
    def conv_op_factory(self, in_channels, out_channels, last):
        activated = not last or self.activated
        return ResBlock(in_channels=in_channels, out_channels=out_channels, 
                             dim=self.dim, activated=activated,
                             **self.res_block_kwargs)

