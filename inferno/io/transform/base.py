from ...utils import python_utils as pyu
import numpy as np


class Transform(object):
    """
    Base class for a Transform. The argument `apply_to` (list) specifies the indices of
    the tensors this transform will be applied to.

    The following methods are recognized (in order of descending priority):
        - `batch_function`: Applies to all tensors in a batch simultaneously
        - `tensor_function`: Applies to just __one__ tensor at a time.
        - `volume_function`: For 3D volumes, applies to just __one__ volume at a time.
        - `image_function`: For 2D or 3D volumes, applies to just __one__ image at a time.

    For example, if both `volume_function` and `image_function` are defined, this means that
    only the former will be called. If the inputs are therefore not 5D batch-tensors of 3D
    volumes, a `NotImplementedError` is raised.
    """
    def __init__(self, apply_to=None):
        """
        Parameters
        ----------
        apply_to : list or tuple
            Indices of tensors to apply this transform to. The indices are with respect
            to the list of arguments this object is called with.
        """
        self._random_variables = {}
        self._apply_to = list(apply_to) if apply_to is not None else None

    def build_random_variables(self, **kwargs):
        pass

    def clear_random_variables(self):
        self._random_variables = {}

    def get_random_variable(self, key, default=None, build=True,
                            **random_variable_building_kwargs):
        if key in self._random_variables:
            return self._random_variables.get(key, default)
        else:
            if not build:
                return default
            else:
                self.build_random_variables(**random_variable_building_kwargs)
                return self.get_random_variable(key, default, build=False)

    def set_random_variable(self, key, value):
        self._random_variables.update({key: value})

    def __call__(self, *tensors, **transform_function_kwargs):
        tensors = pyu.to_iterable(tensors)
        # Get the list of the indices of the tensors to which we're going to apply the transform
        apply_to = list(range(len(tensors))) if self._apply_to is None else self._apply_to
        # Flush random variables and assume they're built by image_function
        self.clear_random_variables()
        if hasattr(self, 'batch_function'):
            transformed = self.batch_function(tensors, **transform_function_kwargs)
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'tensor_function'):
            transformed = [self.tensor_function(tensor, **transform_function_kwargs)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'volume_function'):
            # Loop over all tensors
            transformed = [self._apply_volume_function(tensor, **transform_function_kwargs)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'image_function'):
            # Loop over all tensors
            transformed = [self._apply_image_function(tensor, **transform_function_kwargs)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        else:
            raise NotImplementedError

    # noinspection PyUnresolvedReferences
    def _apply_image_function(self, tensor, **transform_function_kwargs):
        assert pyu.has_callable_attr(self, 'image_function')
        # 2D case
        if tensor.ndim == 4:
            return np.array([np.array([self.image_function(image, **transform_function_kwargs)
                                       for image in channel_image])
                             for channel_image in tensor])
        # 3D case
        elif tensor.ndim == 5:
            return np.array([np.array([np.array([self.image_function(image,
                                                                     **transform_function_kwargs)
                                                 for image in volume])
                                       for volume in channel_volume])
                             for channel_volume in tensor])
        elif tensor.ndim == 3:
            # Assume we have a 3D volume (signature zyx) and apply the image function
            # on all yx slices.
            return np.array([self.image_function(image, **transform_function_kwargs)
                             for image in tensor])
        elif tensor.ndim == 2:
            # Assume we really do have an image.
            return self.image_function(tensor, **transform_function_kwargs)
        else:
            raise NotImplementedError

    # noinspection PyUnresolvedReferences
    def _apply_volume_function(self, tensor, **transform_function_kwargs):
        assert pyu.has_callable_attr(self, 'volume_function')
        # 3D case
        if tensor.ndim == 5:
            # tensor is bczyx
            # volume function is applied to zyx, i.e. loop over b and c
            # FIXME This loops one time too many
            return np.array([np.array([np.array([self.volume_function(volume,
                                                                      **transform_function_kwargs)
                                                 for volume in channel_volume])
                                       for channel_volume in batch])
                             for batch in tensor])
        elif tensor.ndim == 4:
            # We're applying the volume function on a czyx tensor, i.e. we loop over c and apply
            # volume function to (zyx)
            return np.array([self.volume_function(volume, **transform_function_kwargs)
                             for volume in tensor])
        elif tensor.ndim == 3:
            # We're applying the volume function on the volume itself
            return self.volume_function(tensor, **transform_function_kwargs)
        else:
            raise NotImplementedError


class Compose(object):
    """Composes multiple callables (including but not limited to `Transform` objects)."""
    def __init__(self, *transforms):
        """
        Parameters
        ----------
        transforms : list of callable or tuple of callable
            Transforms to compose.
        """
        assert all([callable(transform) for transform in transforms])
        self.transforms = list(transforms)

    def add(self, transform):
        assert callable(transform)
        self.transforms.append(transform)
        return self

    def remove(self, name):
        transform_idx = None
        for idx, transform in enumerate(self.transforms):
            if type(transform).__name__ == name:
                transform_idx = idx
                break
        if transform_idx is not None:
            self.transforms.pop(transform_idx)
        return self

    def __call__(self, *tensors):
        intermediate = tensors
        for transform in self.transforms:
            intermediate = pyu.to_iterable(transform(*intermediate))
        return pyu.from_iterable(intermediate)


class DTypeMapping(object):
    DTYPE_MAPPING = {'float32': 'float32',
                     'float': 'float32',
                     'double': 'float64',
                     'float64': 'float64',
                     'half': 'float16',
                     'float16': 'float16',
                     'long': 'int64',
                     'int64': 'int64',
                     'byte': 'uint8',
                     'uint8': 'uint8',
                     'int': 'int32',
                     'int32': 'int32'}