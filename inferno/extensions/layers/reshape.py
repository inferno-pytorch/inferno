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
