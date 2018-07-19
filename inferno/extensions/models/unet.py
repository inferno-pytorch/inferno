from inferno.extensions.layers.convolutional import ConvELU2D
import torch 
import torch.nn as nn


from .building_blocks import ResBlock, DenseBlock




class UnetBase(nn.Module):
    def __init__(self, in_channels, out_channels=None, ndim, depth=3, gain=2, residual=False, upsample_mode='bilinear'):

        super(UnetBase, self).__init__()
        self.in_channels = in_channels
        self.depth = int(depth)
        self.gain = int(gain)
        self.out_channels = out_channels
        self.residual = residual
        self.ndim = ndim    
        self.upsample_mode = upsample_mode
        if self.out_channels is None:
            self.out_channels = self.in_channels * gain


        conv_in_channels  = in_channels
        # convolution block downwards
        conv_down_ops = []
        for i in range(depth):
            conv = self.conv_down_op_factory( conv_in_channels, int(conv_in_channels*gain))
            conv_in_channels *= gain

            conv_down_ops.append(conv)

        self.conv_down_ops = nn.ModuleList(conv_down_ops)

        # pooling  downsample
        self.downsample_ops = nn.ModuleList([
            self.pooling_op_factory() for i in range(depth)
        ])

        # upsample
        self.upsample_ops = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode=self.upsample_mode) for i in range(depth)
        ])

        # bottom block
        #self.conv_bottom_op = self.conv_bottom_op_factory(conv_in_channels,int(conv_in_channels*gain))
        self.conv_bottom_op = self.conv_bottom_op_factory(conv_in_channels,conv_in_channels)
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
                conv = self.conv_up_op_factory( fac*conv_in_channels, self.out_channels, last=True)
                conv_in_channels //= gain
                conv_up_ops.append(conv)
            else:

                conv = self.conv_up_op_factory( fac*conv_in_channels, conv_in_channels//gain, last=False)
                conv_in_channels //= gain
                conv_up_ops.append(conv)

        self.conv_up_ops = nn.ModuleList(conv_up_ops)

    
    def pooling_op_factory(self):
        if self.ndim == 2:
            return nn.MaxPool2d(kernel_size=2, stride=2) 
        else:
            return nn.MaxPool3d(kernel_size=2, stride=2) 

    def upsample_op_factory(self):
        return nn.Upsample(scale_factor=2, mode=self.upsample_mode)

    def conv_op_factory(self, in_channels, out_channels, last):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def conv_down_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels, last=False)

    def conv_up_op_factory(self, in_channels, out_channels, last):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels, last=last)

    def conv_bottom_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels, last=False)




    def forward(self, input):
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



class ResBlockUnet(UnetBase):
    def __init__(self, in_channels, ndim, out_channels=None, depth=3, gain=2, residual=False, activated=True, **kwargs):
        self.activated = activated
        self.ndim = ndim
        super(ResBlockUnet, self).__init__(
            in_channels=in_channels, 
            ndim=ndim,
            out_channels=out_channels, 
            depth=depth, 
            gain=gain, 
            residual=residual,
            **kwargs
        )

        
    def conv_op_factory(self, in_channels, out_channels, last):
        if self.activated==False and last:
            return ResBlock(in_channels=in_channels, out_channels=out_channels, ndim=self.ndim, activated=False)
        else:
            return ResBlock(in_channels=in_channels, out_channels=out_channels, ndim=self.ndim, activated=True)

# class DenseBlockUnet(UnetBase):
#     def __init__(self, in_channels, ndim,out_channels=None, depth=3, gain=2, residual=False):
#         super(DenseBlockUnet, self).__init__(
#             in_channels=in_channels, 
#             ndim=ndim,
#             out_channels=out_channels, 
#             depth=depth, 
#             gain=gain, 
#             residual=residual
#         )


#     def conv_op_factory(self, in_channels, out_channels):
#         return DenseBlock(in_channels=in_channels, out_channels=out_channels)


