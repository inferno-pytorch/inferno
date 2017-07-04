import numpy as np
import torch
from .base import Transform


class Normalize(Transform):
    """Normalizes input to zero mean unit variance."""
    def __init__(self, eps=1e-4, **super_kwargs):
        """
        Parameters
        ----------
        eps : float
            A small epsilon for numerical stability.
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super(Normalize, self).__init__(**super_kwargs)
        self.eps = eps

    def tensor_function(self, tensor):
        tensor = (tensor - tensor.mean())/(tensor.std() + self.eps)
        return tensor


class NormalizeRange(Transform):
    """Normalizes input by a constant."""
    def __init__(self, normalize_by=255., **super_kwargs):
        """
        Parameters
        ----------
        normalize_by : float or int
            Scalar to normalize by.
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super(NormalizeRange, self).__init__(**super_kwargs)
        self.normalize_by = float(normalize_by)

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
        """
        Parameters
        ----------
        dtype : {'float16', 'float32', 'float64', 'half', 'float', 'double'}
            Datatype to cast to.
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super(Cast, self).__init__(**super_kwargs)
        assert dtype in self.DTYPE_MAPPING.keys()
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def tensor_function(self, tensor):
        return getattr(np, self.dtype)(tensor)


class AsTorchBatch(Transform):
    """Converts a given numpy array to a torch batch tensor.

    The result is a torch tensor __without__ the leading batch axis. For example,
    if the input is an image of shape `(100, 100)`, the output is a batch of shape
    `(1, 100, 100)`. The collate function will add the leading batch axis to obtain
    a tensor of shape `(N, 1, 100, 100)`, where `N` is the batch-size.
    """
    def __init__(self, dimensionality, **super_kwargs):
        """
        Parameters
        ----------
        dimensionality : {1, 2, 3}
            Dimensionality of the data: 1 if vector, 2 if image, 3 if volume.
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super(AsTorchBatch, self).__init__(**super_kwargs)
        assert dimensionality in [1, 2, 3]
        self.dimensionality = dimensionality

    def tensor_function(self, tensor):
        assert isinstance(tensor, np.ndarray)
        if self.dimensionality == 3:
            # We're dealing with a volume. tensor can either be 3D or 4D
            assert tensor.ndim in [3, 4]
            if tensor.ndim == 3:
                # Add channel axis
                return torch.from_numpy(tensor[None, ...])
            else:
                # Channel axis is in already
                return torch.from_numpy(tensor)
        elif self.dimensionality == 2:
            # We're dealing with an image. tensor can either be 2D or 3D
            assert tensor.ndim in [2, 3]
            if tensor.ndim == 2:
                # Add channel axis
                return torch.from_numpy(tensor[None, ...])
            else:
                # Channel axis is in already
                return torch.from_numpy(tensor)
        elif self.dimensionality == 1:
            # We're dealing with a vector - it has to be 1D
            assert tensor.ndim == 1
            return torch.from_numpy(tensor)
        else:
            raise NotImplementedError
