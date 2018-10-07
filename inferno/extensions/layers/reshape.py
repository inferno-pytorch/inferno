import torch
import torch.nn as nn
import torch.nn.functional as F
from ...utils.exceptions import assert_, ShapeError
from ...utils import python_utils as pyu


__all__ = ['View', 'AsMatrix', 'Flatten',
           'As3D', 'As2D',
           'Concatenate', 'Cat',
           'ResizeAndConcatenate', 'PoolCat',
           'GlobalMeanPooling', 'GlobalMaxPooling',
           'Sum', 'SplitChannels','Squeeze', 'RemoveSingletonDimension']
_all = __all__

class View(nn.Module):
    def __init__(self, as_shape):
        super(View, self).__init__()
        self.as_shape = self.validate_as_shape(as_shape)

    def validate_as_shape(self, as_shape):
        assert all([isinstance(_s, int) or _s == 'x' for _s in as_shape])

        all_int_indices = [_n for _n, _s in enumerate(as_shape) if isinstance(_s, int)]
        if all_int_indices:
            first_int_at_index = all_int_indices[0]
            assert all([isinstance(_s, int) for _s in as_shape[first_int_at_index:]])
        return as_shape

    def forward(self, input):
        input_shape = list(input.size())
        reshaped_shape = [_s if isinstance(_s, int) else input_shape[_n]
                          for _n, _s in enumerate(self.as_shape)]
        output = input.view(*reshaped_shape)
        return output


class AsMatrix(View):
    def __init__(self):
        super(AsMatrix, self).__init__(as_shape=['x', 'x'])


class Flatten(View):
    def __init__(self):
        super(Flatten, self).__init__(as_shape=['x', -1])


class As3D(nn.Module):
    def __init__(self, channel_as_z=False, num_channels_or_num_z_slices=1):
        super(As3D, self).__init__()
        self.channel_as_z = channel_as_z
        self.num_channels_or_num_z_slices = num_channels_or_num_z_slices

    def forward(self, input):
        if input.dim() == 5:
            # If input is a batch of 3D volumes - return as is
            return input
        elif input.dim() == 4:
            # If input is a batch of 2D images, reshape
            b, c, _0, _1 = list(input.size())
            assert_(c % self.num_channels_or_num_z_slices == 0,
                    "Number of channels of the 4D image tensor (= {}) must be "
                    "divisible by the set number of channels or number of z slices "
                    "of the 5D volume tensor (= {})."
                    .format(c, self.num_channels_or_num_z_slices),
                    ShapeError)
            c //= self.num_channels_or_num_z_slices
            if self.channel_as_z:
                # Move channel axis to z
                return input.view(b, self.num_channels_or_num_z_slices, c, _0, _1)
            else:
                # Keep channel axis where it is, but add a singleton dimension for z
                return input.view(b, c, self.num_channels_or_num_z_slices, _0, _1)
        elif input.dim() == 2:
            # We have a matrix which we wish to turn to a 3D batch
            b, c = list(input.size())
            return input.view(b, c, 1, 1, 1)
        else:
            raise NotImplementedError


class As2D(nn.Module):
    def __init__(self, z_as_channel=True):
        super(As2D, self).__init__()
        self.z_as_channel = z_as_channel

    def forward(self, input):
        if input.dim() == 5:
            b, c, _0, _1, _2 = list(input.size())
            if not self.z_as_channel:
                assert _0 == 1
            # Reshape
            return input.view(b, c * _0, _1, _2)
        elif input.dim() == 4:
            # Nothing to do here - input is already 2D
            return input
        elif input.dim() == 2:
            # We make singleton dimensions
            b, c = list(input.size())
            return input.view(b, c, 1, 1)


class Concatenate(nn.Module):
    """Concatenate input tensors along a specified dimension."""
    def __init__(self, dim=1):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class ResizeAndConcatenate(nn.Module):
    """
    Resize input tensors spatially (to a specified target size) before concatenating
    them along the a given dim (channel, i.e. 1 by default). The down-sampling mode can
    be specified ('average' or 'max'), but the up-sampling is always 'nearest'.
    """

    POOL_MODE_MAPPING = {'avg': 'avg',
                         'average': 'avg',
                         'mean': 'avg',
                         'max': 'max'}

    def __init__(self, target_size, pool_mode='average', dim=1):
        super(ResizeAndConcatenate, self).__init__()
        self.target_size = target_size
        assert_(pool_mode in self.POOL_MODE_MAPPING.keys(),
                "`pool_mode` must be one of {}, got {} instead."
                .format(self.POOL_MODE_MAPPING.keys(), pool_mode),
                ValueError)
        self.pool_mode = self.POOL_MODE_MAPPING.get(pool_mode)
        self.dim = dim

    def forward(self, *inputs):
        dim = inputs[0].dim()
        assert_(dim in [4, 5],
                'Input tensors must either be 4 or 5 '
                'dimensional, but inputs[0] is {}D.'.format(dim),
                ShapeError)
        # Get resize function
        spatial_dim = {4: 2, 5: 3}[dim]
        resize_function = getattr(F, 'adaptive_{}_pool{}d'.format(self.pool_mode,
                                                                  spatial_dim))
        target_size = pyu.as_tuple_of_len(self.target_size, spatial_dim)
        # Do the resizing
        resized_inputs = []
        for input_num, input in enumerate(inputs):
            # Make sure the dim checks out
            assert_(input.dim() == dim,
                    "Expected inputs[{}] to be a {}D tensor, got a {}D "
                    "tensor instead.".format(input_num, dim, input.dim()),
                    ShapeError)
            resized_inputs.append(resize_function(input, target_size))
        # Concatenate along the channel axis
        if len(resized_inputs) > 1:
            concatenated = torch.cat(tuple(resized_inputs), self.dim)
        else:
            concatenated = resized_inputs[0]
        # Done
        return concatenated


class Cat(Concatenate):
    """An alias for `Concatenate`. Hey, everyone knows who Cat is."""
    pass


class PoolCat(ResizeAndConcatenate):
    """Alias for `ResizeAndConcatenate`, just to annoy snarky web developers."""
    pass


class GlobalMeanPooling(ResizeAndConcatenate):
    """Global mean pooling layer."""
    def __init__(self):
        super(GlobalMeanPooling, self).__init__((1, 1), 'average')


class GlobalMaxPooling(ResizeAndConcatenate):
    """Global max pooling layer."""
    def __init__(self):
        super(GlobalMaxPooling, self).__init__((1, 1), 'max')


class Sum(nn.Module):
    """Sum all inputs."""
    def forward(self, *inputs):
        return torch.stack(inputs, dim=0).sum(0).squeeze(0)


class SplitChannels(nn.Module):
    """Split input at a given index along the channel axis."""
    def __init__(self, channel_index):
        super(SplitChannels, self).__init__()
        self.channel_index = channel_index

    def forward(self, input):
        if isinstance(self.channel_index, int):
            split_location = self.channel_index
        elif self.channel_index == 'half':
            split_location = input.size(1) // 2
        else:
            raise NotImplementedError
        assert split_location < input.size(1)
        split_0 = input[:, 0:split_location, ...]
        split_1 = input[:, split_location:, ...]
        return split_0, split_1



class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    def  forward(self, x):
        return x.squeeze()

class RemoveSingletonDimension(nn.Module):
    def __init__(self, dim=1):
        super(RemoveSingletonDimension, self).__init__()
        self.dim = 1
    def  forward(self, x):
        size = list(x.size())
        if size[self.dim] != 1:
            raise RuntimeError("RemoveSingletonDimension expects a single channel at dim %d, shape=%s"%(self.dim,str(size)))

        slicing = []
        for s in size:
            slicing.append(slice(0, s))

        slicing[self.dim] = 0

        return x[slicing]