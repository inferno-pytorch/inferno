from torch.nn import Module,ModuleList,ELU
from inferno.extensions.layers.convolutional import ConvELU2D, Conv2D,ConvELU3D, Conv3D,ConvActivation
import torch 
import copy
import sys

class ResBlockBase(Module):
    def __init__(self, in_channels, out_channels, dim,  size=2, force_skip_op=False, activated=True):
        super(ResBlockBase, self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.size = int(size)
        self.activated = bool(activated)
        self.force_skip_op = bool(force_skip_op)
        self.dim = int(dim)


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

        if input.size(1) != self.in_channels : 
            raise RuntimeError("wrong number of channels: expected %d, got %d"%
                (self.in_channels, input.size(1))) 

        if input.dim() != self.dim + 2 : 
            raise RuntimeError("wrong number of dim: expected %d, got %d"%
                (self.dim+2, input.dim())) 



        if self.in_channels != self.out_channels or self.force_skip_op:
            skip_res = self.activated_skip_op(input)
        else:
            skip_res = input

        assert skip_res.size(1) == self.out_channels

        res = skip_res
        for i in  range(self.size):
            res = self.conv_ops[i](res)
            assert res.size(1)  == self.out_channels
            if i + 1 < self.size:
                res = self.activation_ops[i](res)

        
        non_activated =  skip_res + res
        if self.activated:
            return self.activation_ops[-1](non_activated)
        else:
            return non_activated



class ResBlock(ResBlockBase):
    def __init__(self, in_channels, out_channels, dim, size=2, activated=True, activation='ReLU', batchnorm=True, force_skip_op=False, conv_kwargs=None):
    
        # trick to store  nn-module before call of super
        # => we put it in a list
        if isinstance(activation, str):
            self.activation_op = [getattr(torch.nn, activation)()]
        elif isinstance(activation, torch.nn.Module):
            self.activation_op = [activation]
        else:
            raise RuntimeError("activation must be a striong or a torch.nn.Module")

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

        self.dim = dim
        self.batchnorm = batchnorm 


        self.conv_1x1_kwargs = dict(
            kernel_size=1, dim=dim, activation=None,
            stride=1, dilation=1, groups=None, depthwise=False, bias=True,
            deconv=False, initialization=None
        )

        super(ResBlock, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            dim=dim,
            size=size, 
            force_skip_op=force_skip_op,
            activated=activated,
        ) 


    def activated_skip_op_factory(self, in_channels, out_channels):
        conv_op = ConvActivation(in_channels=in_channels, out_channels=out_channels, **self.conv_1x1_kwargs)
        if self.batchnorm:
            batchnorm_op = self.batchnorm_op_factory(in_channels=out_channels)
            return torch.nn.Sequential(conv_op, batchnorm_op, self.activation_op[0])
        else:
            return torch.nn.Sequential(conv_op, self.activation_op[0])

    def nonactivated_conv_op_factory(self, in_channels, out_channels, index):
        conv_op = ConvActivation(in_channels=in_channels, out_channels=out_channels, **self.conv_kwargs)
        if self.batchnorm:
            batchnorm_op = self.batchnorm_op_factory(in_channels=out_channels)
            return torch.nn.Sequential(conv_op, batchnorm_op)
        else:
            return conv_op

    def activation_op_factory(self, index):
        return self.activation_op[0]

    def batchnorm_op_factory(self, in_channels):
        bn_cls_name = 'BatchNorm{}d'.format(int(self.dim))
        bn_op_cls = getattr(torch.nn, bn_cls_name)
        return bn_op_cls(in_channels)