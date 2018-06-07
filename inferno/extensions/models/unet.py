from inferno.extensions.layers.convolutional import *

import torch 
import torch.nn as nn



__all__ = ['ResidualUnet']


class ResidualUnet(nn.Module):

    class ConvBlock(nn.Module):
        def __init__(
            self, 
            in_channels, 
            out_channels
        ):
            super(ConvBlock, self).__init__()
            self.a = BNReLUConv2D(kernel_size=3, in_channels=in_channels,   out_channels=out_channels)
            self.b = BNReLUConv2D(kernel_size=3, in_channels=out_channels,  out_channels=out_channels)
            self.c = BNReLUConv2D(kernel_size=3, in_channels=out_channels,  out_channels=out_channels)

        def forward(self, input):
            ra = self.a(input)
            rb = self.b(ra)
            rc = self.c(rb)
            return ra + rc


    def __init__(self, 
            in_channels, depth=3, gain=2,conv_factory = None, 
            conv_down_factory = None,conv_up_factory = ConvBlock, 
            conv_bottom_factory = None):
        """Construct Unet 
         TODO: out channels atm is in_channels*gain
        
        Args:
            in_channels (int): Number of channels of the input tensor
                depth (int, optional): Depth of the unet: A depth of n correspondes to n downsample steps
                gain (int, optional): While going down in the unet, the numbert of channels is multiplied by
                the gain factor. Going up in the unet, the number of channels is divided by the gain.
            conv_factory (None, optional): Factory function which should return a torch.nn.Module.
                conv_factory is only used if neither conv_down_factory, conv_up_factory or conv_bottom_factory
                is specified the conv_factory will be used.
                The default factory will return a residual convolutional block defined in ResidualUnet.ConvBlock
            conv_down_factory (None, optional): Factory function which should return a torch.nn.Module.
                If nothing is specified, conv_factory is used.
            conv_up_factory (None, optional): Factory function which should return a torch.nn.Module.
                If nothing is specified, conv_factory is used.
            conv_bottom_factory (None, optional): Factory function which should return a torch.nn.Module.
                If nothing is specified, conv_factory is used.
        """
        super(ResidualUnet, self).__init__()

        self.depth = int(depth)
        self.gain = int(gain)
        self.out_channels = in_channels * gain


        if self.conv_factory is None:
            self.conv_factory  = self._conv

        if self.conv_down_factory is None:
            self.conv_down_factory  = self.conv_factory

        if self.conv_down_factory is None:
            self.conv_down_factory  = self.conv_factory

        if self.conv_bottom_factory is None:
            self.conv_bottom_factory  = self.conv_factory


        conv_in_channels  = in_channels
        # convolution block downwards
        conv_down_ops = []
        for i in range(depth):
            conv = self._conv_down_factory( conv_in_channels, int(conv_in_channels*gain))
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
        self.conv_bottom_op = self._conv_bottom_factory(conv_in_channels,int(conv_in_channels*gain))
        conv_in_channels *= gain

        # convolution block upwards
        conv_up_ops = []
        for i in range(depth):
            conv = self._conv_up_factory( conv_in_channels, conv_in_channels//gain)
            conv_in_channels //= gain
            conv_up_ops.append(conv)

        self.conv_up_ops = nn.ModuleList(conv_up_ops)

    
    # fallback implementation
    def _conv_factory(self, in_channels, out_channels):
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