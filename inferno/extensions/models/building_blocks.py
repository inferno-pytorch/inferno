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