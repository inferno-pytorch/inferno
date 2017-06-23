import torch.nn as nn


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
    def __init__(self, channel_as_z=False):
        super(As3D, self).__init__()
        self.channel_as_z = channel_as_z

    def forward(self, input):
        if input.dim() == 5:
            # If input is a batch of 3D volumes - return as is
            return input
        elif input.dim() == 4:
            # If input is a batch of 2D images, reshape
            b, c, _0, _1 = list(input.size())
            if self.channel_as_z:
                # Move channel axis to z
                return input.view(b, 1, c, _0, _1)
            else:
                # Keep channel axis where it is, but add a singleton dimension for z
                return input.view(b, c, 1, _0, _1)
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
