from torch.nn import Module,ModuleList,ELU
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D,ConvELU3D, Conv3D
import torch 

class ResBlock(Module):
    def __init__(self, in_channels, out_channels, ndim, size=3, kernel_size=3, activated=True):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        self.activated = activated

        if ndim == 2:
            self.conv_1x1 = ConvELU2D(kernel_size=1, in_channels=in_channels,   out_channels=out_channels)

            conv_ops = [ConvELU2D(kernel_size=kernel_size, in_channels=in_channels,   out_channels=out_channels)]
            for i in range(size-1):
                if i + 1 == size-1: #<- the last
                    conv_ops.append(ConvELU2D(kernel_size=kernel_size, in_channels=out_channels,   out_channels=out_channels))
                else:
                    conv_ops.append(Conv2D(kernel_size=kernel_size, in_channels=out_channels,   out_channels=out_channels))
        elif ndim == 3:
            self.conv_1x1 = ConvELU3D(kernel_size=1, in_channels=in_channels,   out_channels=out_channels)

            conv_ops = [ConvELU3D(kernel_size=kernel_size, in_channels=in_channels,   out_channels=out_channels)]
            for i in range(size-1):
                if i + 1 == size-1: #<- the last
                    conv_ops.append(ConvELU3D(kernel_size=kernel_size, in_channels=out_channels,   out_channels=out_channels))
                else:
                    conv_ops.append(Conv3D(kernel_size=kernel_size, in_channels=out_channels,   out_channels=out_channels))



        self.conv_ops = ModuleList(conv_ops)
        if self.activated:
            self.non_lin = ELU()

    def forward(self, input):
        assert input.size(1) == self.in_channels
        short = self.conv_1x1(input)
        res_list = []
        for i in  range(self.size):
            res = self.conv_ops[i](input)
            res_list.append(res)
            input = res

        assert res_list[0].size(1)  == self.out_channels
        assert res_list[-1].size(1) == self.out_channels
        non_activated =  short + res_list[-1]
        if self.activated:
            return self.non_lin(non_activated)
        else:
            return non_activated

class DenseBlock(Module):
    def __init__(self, in_channels, out_channels,out_per_conv=4, size=4, kernel_size=3):
        super(DenseBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size

        sum_of_channels = int(in_channels)

        conv_ops = []
        for i in range(size):   

            if i + 1 == size:
                conv_ops.append(ConvELU2D(kernel_size=kernel_size, in_channels=sum_of_channels,   out_channels=out_channels))
                sum_of_channels += out_per_conv
            else:
                conv_ops.append(ConvELU2D(kernel_size=kernel_size, in_channels=sum_of_channels,   out_channels=out_per_conv))
                sum_of_channels += out_per_conv

        self.conv_ops = ModuleList(conv_ops)


    def forward(self, input):
        assert input.size(1) == self.in_channels

        for i in  range(self.size):
            res = self.conv_ops[i](input)
            if i + 1 == self.size:
                input = res
            else:
                input = torch.cat([input,res], 1)

        assert input.size(1) == self.out_channels
        return input