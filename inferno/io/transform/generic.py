import numpy as np
import torch
from .base import Transform, DTypeMapping
from ...utils.exceptions import assert_, DTypeError


class Normalize(Transform):
    """Normalizes input to zero mean unit variance."""
    def __init__(self, eps=1e-4, mean=None, std=None, **super_kwargs):
        """
        Parameters
        ----------
        eps : float
            A small epsilon for numerical stability.
        mean : list or float or numpy.ndarray
            Global dataset mean for all channels.
        std : list or float or numpy.ndarray
            Global dataset std for all channels.
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super(Normalize, self).__init__(**super_kwargs)
        self.eps = eps
        self.mean = np.asarray(mean) if mean is not None else None
        self.std = np.asarray(std) if std is not None else None

    def tensor_function(self, tensor):
        mean = np.asarray(tensor.mean()) if self.mean is None else self.mean
        std = np.asarray(tensor.std()) if self.std is None else self.std
        # Figure out how to reshape mean and std
        reshape_as = [-1] + [1] * (tensor.ndim - 1)
        # Normalize
        tensor = (tensor - mean.reshape(*reshape_as))/(std.reshape(*reshape_as) + self.eps)
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


class Project(Transform):
    """
    Given a projection mapping (i.e. a dict) and an input tensor, this transform replaces
    all values in the tensor that equal a key in the mapping with the value corresponding to
    the key.
    """
    def __init__(self, projection, **super_kwargs):
        """
        Parameters
        ----------
        projection : dict
            The projection mapping.
        super_kwargs : dict
            Keywords to the super class.
        """
        super(Project, self).__init__(**super_kwargs)
        self.projection = dict(projection)

    def tensor_function(self, tensor):
        output = np.zeros_like(tensor)
        for source, target in self.projection.items():
            output[tensor == source] = target
        return output


class Label2OneHot(Transform, DTypeMapping):
    """Convert integer labels to one-hot vectors for arbitrary dimensional data."""
    def __init__(self, num_classes, dtype='float', **super_kwargs):
        """
        Parameters
        ----------
        num_classes : int
            Number of classes.
        dtype : str
            Datatype of the output.
        super_kwargs : dict
            Keyword arguments to the superclass.
        """
        super(Label2OneHot, self).__init__(**super_kwargs)
        self.num_classes = num_classes
        self.dtype = self.DTYPE_MAPPING.get(dtype)

    def tensor_function(self, tensor):
        reshaped_arange = np.arange(self.num_classes).reshape(-1, *(1,)*tensor.ndim)
        output = np.equal(reshaped_arange, tensor).astype(self.dtype)
        # output = np.zeros(shape=(self.num_classes,) + tensor.shape, dtype=self.dtype)
        # # Optimizing for simplicity and memory efficiency, because one would usually
        # # spawn multiple workers
        # for class_num in range(self.num_classes):
        #     output[class_num] = tensor == class_num
        return output


class Cast(Transform, DTypeMapping):
    """Casts inputs to a specified datatype."""
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
    def __init__(self, dimensionality, add_channel_axis_if_necessary=True, **super_kwargs):
        """
        Parameters
        ----------
        dimensionality : {1, 2, 3}
            Dimensionality of the data: 1 if vector, 2 if image, 3 if volume.
        add_channel_axis_if_necessary : bool
            Whether to add a channel axis where necessary. For example, if `dimensionality = 2`
            and the input temperature has 2 dimensions (i.e. an image), setting
            `add_channel_axis_if_necessary` to True results in the output being a 3 dimensional
            tensor, where the leading dimension is a singleton and corresponds to `channel`.
        super_kwargs : dict
            Kwargs to the superclass `inferno.io.transform.base.Transform`.
        """
        super(AsTorchBatch, self).__init__(**super_kwargs)
        assert dimensionality in [1, 2, 3]
        self.dimensionality = dimensionality
        self.add_channel_axis_if_necessary = bool(add_channel_axis_if_necessary)

    def _to_batch(self, tensor):
        assert_(isinstance(tensor, np.ndarray),
                "Expected numpy array, got %s" % type(tensor),
                DTypeError)
        if self.dimensionality == 3:
            # We're dealing with a volume. tensor can either be 3D or 4D
            assert tensor.ndim in [3, 4]
            if tensor.ndim == 3 and self.add_channel_axis_if_necessary:
                # Add channel axis
                return torch.from_numpy(tensor[None, ...])
            else:
                # Channel axis is in already
                return torch.from_numpy(tensor)
        elif self.dimensionality == 2:
            # We're dealing with an image. tensor can either be 2D or 3D
            assert tensor.ndim in [2, 3]
            if tensor.ndim == 2 and self.add_channel_axis_if_necessary:
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

    def tensor_function(self, tensor):
        assert_(isinstance(tensor, (list, np.ndarray)),
                "Expected numpy array or list, got %s" % type(tensor),
                DTypeError)
        if isinstance(tensor, np.ndarray):
            return self._to_batch(tensor)
        else:
            return [self._to_batch(elem) for elem in tensor]
