from inferno.extensions.layers.convolutional import ConvELU2D
import torch 
import torch.nn as nn


from building_blocks import *




class UnetBase(nn.Module):
    def __init__(self, in_channels, out_channels=None, depth=3, gain=2, residual=False,ndim=2, p_dropout=0.1, upsample_mode='bilinear'):

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
            nn.Upsample(scale_factor=2, mode='bilinear') for i in range(depth)
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

            if i + 1 == depth:
                conv = self.conv_up_op_factory( fac*conv_in_channels, self.out_channels)
                conv_in_channels //= gain
                conv_up_ops.append(conv)
            else:

                conv = self.conv_up_op_factory( fac*conv_in_channels, conv_in_channels//gain)
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

    def conv_op_factory(self, in_channels, out_channels):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def conv_down_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels)

    def conv_up_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels)

    def conv_bottom_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels)




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
    def __init__(self, in_channels, out_channels=None, depth=3, gain=2, residual=False):
        super(ResBlockUnet, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            depth=depth, 
            gain=gain, 
            residual=residual
        )


    def conv_op_factory(self, in_channels, out_channels):
        return ResBlock(in_channels=in_channels, out_channels=out_channels)


class DenseBlockUnet(UnetBase):
    def __init__(self, in_channels, out_channels=None, depth=3, gain=2, residual=False):
        super(DenseBlockUnet, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            depth=depth, 
            gain=gain, 
            residual=residual
        )


    def conv_op_factory(self, in_channels, out_channels):
        return DenseBlock(in_channels=in_channels, out_channels=out_channels)from inferno.extensions.layers.convolutional import ConvELU2D
import torch 
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size=3, kernel_size=3):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size


        conv_ops = [ConvELU2D(kernel_size=kernel_size, in_channels=in_channels,   out_channels=out_channels)]
        for i in range(size-1):
            conv_ops.append(ConvELU2D(kernel_size=kernel_size, in_channels=out_channels,   out_channels=out_channels))

        self.conv_ops = nn.ModuleList(conv_ops)


    def forward(self, input):
        assert input.size(1) == self.in_channels

        res_list = []
        for i in  range(self.size):
            res = self.conv_ops[i](input)
            res_list.append(res)
            input = res

        assert res_list[0].size(1)  == self.out_channels
        assert res_list[-1].size(1) == self.out_channels
        return res_list[0] + res_list[-1]

class DenseBlock(nn.Module):
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

        self.conv_ops = nn.ModuleList(conv_ops)


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

class UnetBase(nn.Module):
    def __init__(self, in_channels, out_channels=None, depth=3, gain=2, residual=False, p_dropout=0.1):

        super(UnetBase, self).__init__()
        self.in_channels = in_channels
        self.depth = int(depth)
        self.gain = int(gain)
        self.out_channels = out_channels
        self.residual = residual
        if self.out_channels is None:
            self.out_channels = self.in_channels * gain

        self.dropout = torch.nn.Dropout2d(p=p_dropout)

        conv_in_channels  = in_channels
        # convolution block downwards
        conv_down_ops = []
        for i in range(depth):
            conv = self.conv_down_op_factory( conv_in_channels, int(conv_in_channels*gain))
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

            if i + 1 == depth:
                conv = self.conv_up_op_factory( fac*conv_in_channels, self.out_channels)
                conv_in_channels //= gain
                conv_up_ops.append(conv)
            else:

                conv = self.conv_up_op_factory( fac*conv_in_channels, conv_in_channels//gain)
                conv_in_channels //= gain
                conv_up_ops.append(conv)

        self.conv_up_ops = nn.ModuleList(conv_up_ops)

    
    def conv_op_factory(self, in_channels, out_channels):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def conv_down_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels)

    def conv_up_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels)

    def conv_bottom_op_factory(self, in_channels, out_channels):
        return self.conv_op_factory(in_channels=in_channels,  out_channels=out_channels)


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
    def __init__(self, in_channels, out_channels=None, depth=3, gain=2, residual=False):
        super(ResBlockUnet, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            depth=depth, 
            gain=gain, 
            residual=residual
        )


    def conv_op_factory(self, in_channels, out_channels):
        return ResBlock(in_channels=in_channels, out_channels=out_channels)


class DenseBlockUnet(UnetBase):
    def __init__(self, in_channels, out_channels=None, depth=3, gain=2, residual=False):
        super(DenseBlockUnet, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            depth=depth, 
            gain=gain, 
            residual=residual
        )


    def conv_op_factory(self, in_channels, out_channels):
        return DenseBlock(in_channels=in_channels, out_channels=out_channels)