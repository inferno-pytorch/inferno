from ...utils import python_utils as pyu
import numpy as np
import logging


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
        self.logger = logging.getLogger("{}-@-{}".format(type(self).__name__, id(self)))

    def build_random_variables(self, **kwargs):
        pass

    def clear_random_variables(self):
        self._random_variables = {}

    def get_random_variable(self, key, default=None, build=True,
                            **random_variable_building_kwargs):
        if key in self._random_variables:
            self.logger.debug("Fetching random variable: {}.".format(key))
            return self._random_variables.get(key, default)
        else:
            if not build:
                self.logger.debug("Tried to fetch random variable {}, "
                                  "but failed. Not allowed to build, "
                                  "so returning default of type {}."
                                  .format(key, type(default).__name__))
                return default
            else:
                self.logger.debug("Tried to fetch random variable {}, "
                                  "but failed; building one.".format(key))
                self.build_random_variables(**random_variable_building_kwargs)
                self.logger.debug("Random variable {} built.".format(key))
                return self.get_random_variable(key, default, build=False)

    def set_random_variable(self, key, value):
        self.logger.debug("Setting random variable {} to a {}."
                          .format(key, type(value).__name__))
        self._random_variables.update({key: value})

    def __call__(self, *tensors):
        tensors = pyu.to_iterable(tensors)
        self.logger.debug("Calling on {} tensors.".format(len(tensors)))
        # Get the list of the indices of the tensors to which we're going to apply the transform
        apply_to = list(range(len(tensors))) if self._apply_to is None else self._apply_to
        self.logger.debug("Applying transform to tensors indexed: {}".format(apply_to))
        # Flush random variables and assume they're built by image_function
        self.clear_random_variables()
        self.logger.debug("Cleared random variables.")
        if hasattr(self, 'batch_function'):
            self.logger.debug("Applying transform with batch_function.")
            transformed = self.batch_function(tensors)
            self.logger.debug("Applied transform successfully")
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'tensor_function'):
            self.logger.debug("Applying transform with tensor_function.")
            transformed = [getattr(self, 'tensor_function')(tensor)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            self.logger.debug("Applied transform successfully")
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'volume_function'):
            self.logger.debug("Applying transform with volume_function.")
            # Loop over all tensors
            transformed = [self._apply_volume_function(tensor)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            self.logger.debug("Applied transform successfully")
            return pyu.from_iterable(transformed)
        elif hasattr(self, 'image_function'):
            self.logger.debug("Applying transform with image_function.")
            # Loop over all tensors
            transformed = [self._apply_image_function(tensor)
                           if tensor_index in apply_to else tensor
                           for tensor_index, tensor in enumerate(tensors)]
            self.logger.debug("Applied transform successfully")
            return pyu.from_iterable(transformed)
        else:
            self.logger.error("Could not find transform application method.")
            raise NotImplementedError

    def _apply_image_function(self, tensor):
        # 2D case
        if tensor.ndim == 4:
            self.logger.debug("Applying image_function to a 4D tensor.")
            return np.array([np.array([getattr(self, 'image_function')(image)
                                       for image in channel_image])
                             for channel_image in tensor])
        # 3D case
        elif tensor.ndim == 5:
            self.logger.debug("Applying image_function to a 5D tensor.")
            return np.array([np.array([np.array([getattr(self, 'image_function')(image)
                                                 for image in volume])
                                       for volume in channel_volume])
                             for channel_volume in tensor])
        elif tensor.ndim == 3:
            self.logger.debug("Applying image_function to a 3D tensor.")
            # Assume we have a 3D volume (signature zyx) and apply the image function
            # on all yx slices.
            return np.array([getattr(self, 'image_function')(image) for image in tensor])
        elif tensor.ndim == 2:
            self.logger.debug("Applying image_function to a 2D tensor (image).")
            # Assume we really do have an image.
            return getattr(self, 'image_function')(tensor)
        else:
            self.logger.error("Could not apply image_function to a {}-D tensor."
                              .format(tensor.ndim))
            raise NotImplementedError

    def _apply_volume_function(self, tensor):
        # 3D case
        if tensor.ndim == 5:
            self.logger.debug("Applying volume_function to a 5D tensor.")
            return np.array([np.array([np.array([getattr(self, 'volume_function')(volume)
                                                 for volume in channel_volume])
                                       for channel_volume in batch])
                             for batch in tensor])
        elif tensor.ndim == 4:
            self.logger.debug("Applying volume_function to a 4D tensor.")
            # We're applying the volume function on a czyx tensor
            return np.array([getattr(self, 'volume_function')(volume)
                             for volume in tensor])
        elif tensor.ndim == 3:
            self.logger.debug("Applying volume_function to a 3D tensor.")
            # We're applying the volume function on the volume itself
            return getattr(self, 'volume_function')(tensor)
        else:
            self.logger.error("Could not apply volume_function to a {}-D tensor."
                              .format(tensor.ndim))
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
        self.logger = logging.getLogger("{}-@-{}".format(type(self).__name__, id(self)))
        self.transforms = list(transforms)
        self.logger.debug("Composing loggers: {}"
                          .format([self.get_transform_name(transform)
                                   for transform in self.transforms]))

    @staticmethod
    def get_transform_name(transform):
        return type(transform).__name__

    def add(self, transform):
        assert callable(transform)
        self.logger.debug("Appending transform {} to composition."
                          .format(self.get_transform_name(transform)))
        self.transforms.append(transform)
        return self

    def __call__(self, *tensors):
        intermediate = tensors
        for transform in self.transforms:
            self.logger.debug("Feeding transform {} {} tensors."
                              .format(self.get_transform_name(transform),
                                      len(intermediate)))
            intermediate = pyu.to_iterable(transform(*intermediate))
            self.logger.debug("Obtained {} tensors from transform {}."
                              .format(len(intermediate),
                                      self.get_transform_name(transform)))
        return pyu.from_iterable(intermediate)
