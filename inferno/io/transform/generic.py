import numpy as np
from .base import Transform


class Normalize(Transform):
    def __init__(self, eps=1e-4, **super_kwargs):
        """Normalizes input to zero mean unit variance."""
        super(Normalize, self).__init__(**super_kwargs)
        self.eps = eps

    def tensor_function(self, tensor):
        tensor = (tensor - tensor.mean())/(tensor.std() + self.eps)
        return tensor


class NormalizeRange(Transform):
    """Normalizes input by a constant."""
    def __init__(self, normalize_by=255., **super_kwargs):
        super(NormalizeRange, self).__init__(**super_kwargs)
        self.normalize_by = normalize_by

    def tensor_function(self, tensor):
        return tensor / self.normalize_by


class Cast(Transform):
    """Casts inputs to a specified datatype."""
    DTYPE_MAPPING = {'float32': 'float32',
                     'float': 'float32',
                     'double': 'float64',
                     'float64': 'float64',
                     'half': 'float16',
                     'float16': 'float16'}

    def __init__(self, dtype='float', **super_kwargs):
        super(Cast, self).__init__(**super_kwargs)
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def tensor_function(self, tensor):
        return getattr(np, self.dtype)(tensor)
