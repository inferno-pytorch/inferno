from torch.nn import Module,ModuleList,ELU
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D,ConvELU3D, Conv3D,ConvActivation
import torch 
import copy

class ResBlockBase(Module):
    def __init__(self, in_channels, out_channels, size=2, force_skip_op=False, activated=True):
        super(ResBlockBase, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.size = int(size)
        self.activated = bool(activated)
        self.force_skip_op = bool(force_skip_op)



    def build_model(self):

        if self.in_channels != self.out_channels or self.force_skip_op:
            self.activated_skip_op = self.activated_skip_op_factory(in_channels=self.in_channels, out_channels=self.out_channels)

        conv_ops = []
        activation_ops = []
        for i in range(self.size):

            # the convolutions
            if i == 0:
                op = self.nonactivated_conv_op_factory(in_channels=self.out_channels, out_channels=self.out_channels, index=i)
            else:
                op = self.nonactivated_conv_op_factory(in_channels=self.out_channels, out_channels=self.out_channels, index=i)
            conv_ops.append(op)

            # the activations
            if i<self.size or self.activated:
                activation_ops.append(self.activation_op_factory(index=i))

        self.conv_ops = torch.nn.ModuleList(conv_ops)
        self.activation_ops = torch.nn.ModuleList(activation_ops)


    def activated_skip_op_factory(self, in_channels, out_channels):
        raise NotImplementedError("activated_skip_op_factory need to be implemented by deriving class")

    def nonactivated_conv_op_factory(self, in_channels, out_channels, index):
        raise NotImplementedError("conv_op_factory need to be implemented by deriving class")

    def activation_op_factory(self, index):
        return nn.ReLU()

    def forward(self, input):
        assert input.size(1) == self.in_channels

        if self.in_channels != self.out_channels or self.force_skip_op:
            skip_res = self.activated_skip_op(input)
        else:
            skip_res = input

        assert skip_res.size(1) == self.out_channels

        res = skip_res
        for i in  range(self.size):
            res = self.conv_ops[i](res)
            assert res.size(1)  == self.out_channels
            if i < self.size:
                res = self.activation_ops[i](res)
            input = res

        
        non_activated =  skip_res + res
        if self.activated:
            return self.activation_ops[-1](non_activated)
        else:
            return non_activated



class ResBlock(ResBlockBase):
    def __init__(self, in_channels, out_channels, dim, size=2, activated=True, activation='ReLU', batchnorm=True, force_skip_op=False, conv_kwargs=None):
        
        super(ResBlock, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            size=size, 
            force_skip_op=force_skip_op,
            activated=activated,
        ) 

        self.dim = dim
        self.batchnorm = batchnorm 



        # keywords for conv
        if conv_kwargs is None:
            conv_kwargs = dict(
                 kernel_size=3, dim=dim, activation=None,
                 stride=1, dilation=1, groups=None, depthwise=False, bias=True,
                 deconv=False, initialization=None
            )
        elif isinstance(conv_kwargs, dict):
            conv_kwargs['activation'] = None
        else:
            raise RuntimeError("conv_kwargs must be either None or a dict")
        self.conv_kwargs = conv_kwargs
   

        self.conv_1x1_kwargs = dict(
            kernel_size=1, dim=dim, activation=None,
            stride=1, dilation=1, groups=None, depthwise=False, bias=True,
            deconv=False, initialization=None
        )

        

        if isinstance(activation, str):
            self.activation_op = getattr(torch.nn, activation)()
        elif isinstance(activation, torch.nn.Module):
            self.activation_op = activation
        else:
            raise RuntimeError("activation must be a striong or a torch.nn.Module")

        # we cannot build the model directly 
        # in the constructor of the base class
        # since :
        # - super(ResBlock, self).__init__() needs
        #   to be called before any assignments of submodules:
        #   smth. like `self.activation = kwargs['activation']`
        #   works only after the superinit 
        # - the factory functions like activated_skip_op_factory \
        #   need to access the submodules =>
        #   therefore superinit needs to be called before
        #   any call of ***_factory.
        # - The _factories also need to call members
        #   which only exist in this deriving class,
        #   and those member can be nn-modules. => these 
        #   are not yet ready in super(ResBlock, self).__init__() call
        #   since we HAVE to make these assignments after the superinit 
        # - => this is why we need this build_model 
        #   function
        self.build_model()


    def activated_skip_op_factory(self, in_channels, out_channels):
        conv_op = ConvActivation(in_channels=in_channels, out_channels=out_channels, **self.conv_1x1_kwargs)
        if self.batchnorm:
            batchnorm_op = self.batchnorm_op_factory(in_channels=out_channels)
            return torch.nn.Sequential(conv_op, batchnorm_op, self.activation_op)
        else:
            return torch.nn.Sequential(conv_op, self.activation_op)

    def nonactivated_conv_op_factory(self, in_channels, out_channels, index):
        conv_op = ConvActivation(in_channels=in_channels, out_channels=out_channels, **self.conv_kwargs)
        if self.batchnorm:
            batchnorm_op = self.batchnorm_op_factory(in_channels=out_channels)
            return torch.nn.Sequential(conv_op, batchnorm_op)
        else:
            return conv_op

    def activation_op_factory(self, index):
        return self.activation_op

    def batchnorm_op_factory(self, in_channels):
        bn_cls_name = 'BatchNorm{}d'.format(int(self.dim))
        bn_op_cls = getattr(torch.nn, bn_cls_name)
        return bn_op_cls(in_channels)











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