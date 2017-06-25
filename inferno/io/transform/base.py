from ...utils import python_utils as pyu
import numpy as np


class Transform(object):
    def __init__(self, apply_to=None):
        """
        Base class for a Transform. The argument `apply_to` (list) specifies the indices of
        the tensors this transform will be applied to.
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


class Compose(object):
    """Composes multiple callables (including `Transform`s)."""
    def __init__(self, *transforms):
        assert all([callable(transform) for transform in transforms])
        self.transforms = list(transforms)

    def __call__(self, *tensors):
        intermediate = tensors
        for transform in self.transforms:
            intermediate = pyu.to_iterable(transform(*intermediate))
        return pyu.from_iterable(intermediate)
