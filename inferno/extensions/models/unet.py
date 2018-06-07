from inferno.extensions.layers.convolutional import *

import torch 
import torch.nn as nn



__all__ = ['ResidualUnet',
           'ConvELU2D', 'ConvELU3D',
           'ConvSigmoid2D', 'ConvSigmoid3D',
           'DeconvELU2D', 'DeconvELU3D',
           'StridedConvELU2D', 'StridedConvELU3D',
           'DilatedConvELU2D', 'DilatedConvELU3D',
           'Conv2D', 'Conv3D',
           'BNReLUConv2D', 'BNReLUConv3D',
           'BNReLUDepthwiseConv2D',
           'ConvSELU2D', 'ConvSELU3D']
           

class ResidualUnet(nn.Module):



    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ConvBlock, self).__init__()
            self.a = BNReLUConv2D(kernel_size=3, in_channels=in_channels,   out_channels=out_channels)
            self.b = BNReLUConv2D(kernel_size=3, in_channels=out_channels,  out_channels=out_channels)
            self.c = BNReLUConv2D(kernel_size=3, in_channels=out_channels,  out_channels=out_channels)

        def forward(self, input):
            ra = self.a(input)
            rb = self.b(ra)
            rc = self.c(rb)
            return ra + rc



    def __init__(self, in_channels, depth=3, gain=2):

        super(ResidualUnet, self).__init__()

        self.depth = int(depth)
        self.gain = int(gain)
        self.out_channels = in_channels * gain

        conv_in_channels  = in_channels
        # convolution block downwards
        conv_down_ops = []
        for i in range(depth):
            conv = self.conv_down_ops_factory( conv_in_channels, int(conv_in_channels*gain))
            conv_in_channels *= gain

            conv_down_ops.append(conv)

        self.conv_down_ops = nn.ModuleList(conv_down_ops)

        # downsample
        self.downsample_ops = nn.ModuleList([
            nn.MaxPool2d(kernel_size=2, stride=2) for i in range(depth)
        ])

        # upsample
        self.upsample_ops = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear') for i in range(depth)
        ])

        # bottom block
        self.conv_bottom_op = self.conv_bottom_ops_factory(conv_in_channels,int(conv_in_channels*gain))
        conv_in_channels *= gain

        # convolution block upwards
        conv_up_ops = []
        for i in range(depth):
            conv = self.conv_up_ops_factory( conv_in_channels, conv_in_channels//gain)
            conv_in_channels //= gain
            conv_up_ops.append(conv)

        self.conv_up_ops = nn.ModuleList(conv_up_ops)

    
    def conv_down_ops_factory(self, in_channels, out_channels):
        return ConvBlock(in_channels=in_channels,  out_channels=out_channels)

    def conv_up_ops_factory(self, in_channels, out_channels):
        return ConvBlock(in_channels=in_channels,  out_channels=out_channels)

    def conv_bottom_ops_factory(self, in_channels, out_channels):
        return ConvBlock(in_channels=in_channels,  out_channels=out_channels)




    def forward(self, input):
        down_res = []

        # downwards
        for d in range(self.depth):

            res = self.conv_down_ops[d](input)
            down_res.append(res)
            input = self.downsample_ops[d](res)
        
        input = self.conv_bottom_op(input)

        down_res = list(reversed(down_res))

        # upwards
        for d in range(self.depth):

            # upsample
            input = self.upsample_ops[d](input)
    
            # conv
            input = self.conv_up_ops[d](input)

            a = down_res[d]

            input = a + input
            
        return input 