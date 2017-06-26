from ...utils import python_utils as pyu
import numpy as np


class Transform(object):
    def __init__(self, apply_to=None):
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

    def __call__(self, *tensors):
        tensors = pyu.to_iterable(tensors)
        # Get the list of the indices of the tensors to which we're going to apply the transform
        apply_to = list(range(len(tensors))) if self._apply_to is None else self._apply_to
        # Flush random variables and assume they're built by image_function
        self.clear_random_variables()
        if hasattr(self, 'batch_function'):
            transformed = self.batch_function(tensors)
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'tensor_function'):
            transformed = [getattr(self, 'tensor_function')(tensor)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'volume_function'):
            # Loop over all tensors
            transformed = [self._apply_volume_function(tensor)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'image_function'):
            # Loop over all tensors
            transformed = [self._apply_image_function(tensor)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            return pyu.from_iterable(transformed)
        else:
            raise NotImplementedError

    def _apply_image_function(self, tensor):
        # 2D case
        if tensor.ndim == 4:
            return np.array([np.array([getattr(self, 'image_function')(image)
                                       for image in channel_image])
                             for channel_image in tensor])
        # 3D case
        elif tensor.ndim == 5:
            return np.array([np.array([np.array([getattr(self, 'image_function')(image)
                                                 for image in volume])
                                       for volume in channel_volume])
                             for channel_volume in tensor])
        elif tensor.ndim == 3:
            # Assume we have a 3D volume (signature zyx) and apply the image function
            # on all yx slices.
            return np.array([getattr(self, 'image_function')(image) for image in tensor])
        elif tensor.ndim == 2:
            # Assume we really do have an image.
            return getattr(self, 'image_function')(tensor)
        else:
            raise NotImplementedError

    def _apply_volume_function(self, tensor):
        # 3D case
        if tensor.ndim == 5:
            return np.array([np.array([np.array([getattr(self, 'volume_function')(volume)
                                                 for volume in channel_volume])
                                       for channel_volume in batch])
                             for batch in tensor])
        elif tensor.ndim == 3:
            # We're applying the volume function on the volume itself
            return getattr(self, 'volume_function')(tensor)
        else:
            raise NotImplementedError


class Compose(object):
    """Composes multiple callables (including but not limited to `Transform` objects)."""
    def __init__(self, *transforms):
        assert all([callable(transform) for transform in transforms])
        self.transforms = list(transforms)

    def __call__(self, *tensors):
        intermediate = tensors
        for transform in self.transforms:
            intermediate = pyu.to_iterable(transform(*intermediate))
        return pyu.from_iterable(intermediate)
