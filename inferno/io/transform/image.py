import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage.exposure import adjust_gamma
from warnings import catch_warnings, simplefilter

from .base import Transform
from ...utils.exceptions import assert_, ShapeError


class PILImage2NumPyArray(Transform):
    """Convert a PIL Image object to a numpy array.

    For images with multiple channels (say RGB), the channel axis is moved to front. Therefore,
    a (100, 100, 3) RGB image becomes an array of shape (3, 100, 100).
    """
    def tensor_function(self, tensor):
        tensor = np.asarray(tensor)
        if tensor.ndim == 3:
            # There's a channel axis - we move it to front
            tensor = np.moveaxis(tensor, source=-1, destination=0)
        elif tensor.ndim == 2:
            pass
        else:
            raise NotImplementedError("Expected tensor to be a 2D or 3D "
                                      "numpy array, got a {}D array instead."
                                      .format(tensor.ndim))
        return tensor


class Scale(Transform):
    """Scales an image to a given size with spline interpolation of requested order.

    Unlike torchvision.transforms.Scale, this does not depend on PIL and therefore works
    with numpy arrays. If you do have a PIL image and wish to use this transform, consider
    applying `PILImage2NumPyArray` first.

    Warnings
    --------
    This transform uses `scipy.ndimage.zoom` and requires scipy >= 0.13.0 to work correctly.
    """
    def __init__(self, output_image_shape, interpolation_order=3, zoom_kwargs=None, **super_kwargs):
        """
        Parameters
        ----------
        output_image_shape : list or tuple or int
            Target size of the output image. Aspect ratio may not be preserved.
        interpolation_order : int
            Interpolation order for the spline interpolation.
        zoom_kwargs : dict
            Keyword arguments for `scipy.ndimage.zoom`.
        super_kwargs : dict
            Keyword arguments for the superclass.
        """
        super(Scale, self).__init__(**super_kwargs)
        output_image_shape = (output_image_shape, output_image_shape) \
            if isinstance(output_image_shape, int) else tuple(output_image_shape)
        assert_(len(output_image_shape) == 2,
                "`output_image_shape` must be an integer or a tuple of length 2.",
                ValueError)
        self.output_image_shape = output_image_shape
        self.interpolation_order = interpolation_order
        self.zoom_kwargs = {} if zoom_kwargs is None else dict(zoom_kwargs)

    def image_function(self, image):
        source_height, source_width = image.shape
        target_height, target_width = self.output_image_shape
        # We're on Python 3 - take a deep breath and relax.
        zoom_height, zoom_width = (target_height / source_height), (target_width / source_width)
        with catch_warnings():
            # Ignore warning that scipy should be > 0.13 (it's 0.19 these days)
            simplefilter('ignore')
            rescaled_image = zoom(image, (zoom_height, zoom_width),
                                  order=self.interpolation_order, **self.zoom_kwargs)
        # This should never happen
        assert_(rescaled_image.shape == (target_height, target_width),
                "Shape mismatch that shouldn't have happened if you were on scipy > 0.13.0. "
                "Are you on scipy > 0.13.0?",
                ShapeError)
        return rescaled_image


class RandomCrop(Transform):
    """Crop input to a given size.

    This is similar to torchvision.transforms.RandomCrop, except that it operates on
    numpy arrays instead of PIL images. If you do have a PIL image and wish to use this
    transform, consider applying `PILImage2NumPyArray` first.

    Warnings
    --------
    If `output_image_shape` is larger than the image itself, the image is not cropped
    (along the relevant dimensions).
    """
    def __init__(self, output_image_shape, **super_kwargs):
        """
        Parameters
        ----------
        output_image_shape : tuple or list or int
            Expected shape of the output image. Could be an integer, (say) 100, in
            which case it's interpreted as `(100, 100)`. Note that if the image shape
            along some (or all) dimension is smaller, say `(50, 200)`, the resulting
            output images will have the shape `(50, 100)`.
        super_kwargs : dict
            Keywords to the super class.
        """
        super(RandomCrop, self).__init__(**super_kwargs)
        # Privates
        self._image_shape_cache = None
        # Publics
        output_image_shape = (output_image_shape, output_image_shape) \
            if isinstance(output_image_shape, int) else tuple(output_image_shape)
        assert_(len(output_image_shape) == 2,
                "`output_image_shape` must be an integer or a tuple of length 2.",
                ValueError)
        self.output_image_shape = output_image_shape

    def clear_random_variables(self):
        self._image_shape_cache = None
        super(RandomCrop, self).clear_random_variables()

    def build_random_variables(self, height_leeway, width_leeway):
        self.set_random_variable('height_location',
                                 np.random.randint(low=0, high=height_leeway + 1))
        self.set_random_variable('width_location',
                                 np.random.randint(low=0, high=width_leeway + 1))

    def image_function(self, image):
        # Validate image shape
        if self._image_shape_cache is not None:
            assert_(self._image_shape_cache == image.shape,
                    "RandomCrop works on multiple images simultaneously only "
                    "if they have the same shape. Was expecting an image of "
                    "shape {}, got one of shape {} instead."
                    .format(self._image_shape_cache, image.shape),
                    ShapeError)
        else:
            self._image_shape_cache = image.shape
        source_height, source_width = image.shape
        crop_height, crop_width = self.output_image_shape
        height_leeway = source_height - crop_height
        width_leeway = source_width - crop_width
        if height_leeway > 0:
            # Crop height
            height_location = self.get_random_variable('height_location',
                                                       height_leeway=height_leeway,
                                                       width_leeway=width_leeway)
            cropped = image[height_location:(height_location + crop_height), :]
        else:
            cropped = image
        if width_leeway > 0:
            # Crop width
            width_location = self.get_random_variable('width_location',
                                                      height_leeway=height_leeway,
                                                      width_leeway=width_leeway)
            cropped = cropped[:, width_location:(width_location + crop_width)]
        assert cropped.shape == self.output_image_shape, "Well, shit."
        return cropped


class RandomSizedCrop(Transform):
    """Extract a randomly sized crop from the image.

    The ratio of the sizes of the cropped and the original image can be limited within
    specified bounds along both axes. To resize back to a constant sized image, compose
    with `Scale`.
    """
    def __init__(self, ratio_between=None, height_ratio_between=None, width_ratio_between=None,
                 preserve_aspect_ratio=False, relative_target_aspect_ratio=None, **super_kwargs):
        """
        Parameters
        ----------
        ratio_between : tuple
            Specify the bounds between which to sample the crop ratio. This applies to
            both height and width if not overriden. Can be None if both height and width
            ratios are specified individually.
        height_ratio_between : tuple
            Specify the bounds between which to sample the vertical crop ratio.
            Can be None if `ratio_between` is not None.
        width_ratio_between : tuple
            Specify the bounds between which to sample the horizontal crop ratio.
            Can be None if `ratio_between` is not None.
        preserve_aspect_ratio : bool
            Whether to preserve aspect ratio. If both `height_ratio_between`
            and `width_ratio_between` are specified, the former is used if this
            is set to True.
        relative_target_aspect_ratio : float
            Specify the target aspect ratio (W x H) relative to the input image
            (i.e. by mapping the input image ratio to 1:1). For instance, if an image
            has the size 1024 (H) x 2048 (W), a relative target aspect ratio of 0.5
            might yield images of size 1024 x 1024. Note that this only applies if
            `preserve_aspect_ratio` is set to False.
        super_kwargs : dict
            Keyword arguments for the super class.
        """
        super(RandomSizedCrop, self).__init__(**super_kwargs)
        # Privates
        self._image_shape_cache = None
        # Publics
        height_ratio_between = tuple(height_ratio_between) \
            if height_ratio_between is not None else tuple(ratio_between)
        width_ratio_between = tuple(width_ratio_between) \
            if width_ratio_between is not None else tuple(ratio_between)
        assert_(height_ratio_between is not None,
                "`height_ratio_between` is not specified.",
                ValueError)
        assert_(width_ratio_between is not None,
                "`width_ratio_between` is not specified.",
                ValueError)
        self.height_ratio_between = height_ratio_between
        self.width_ratio_between = width_ratio_between
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.relative_target_aspect_ratio = relative_target_aspect_ratio

    def build_random_variables(self, image_shape):
        # Seed RNG
        np.random.seed()
        # Compute random variables
        source_height, source_width = image_shape
        height_ratio = np.random.uniform(low=self.height_ratio_between[0],
                                         high=self.height_ratio_between[1])
        if self.preserve_aspect_ratio:
            width_ratio = height_ratio
        elif self.relative_target_aspect_ratio is not None:
            width_ratio = height_ratio * self.relative_target_aspect_ratio
        else:
            width_ratio = np.random.uniform(low=self.width_ratio_between[0],
                                            high=self.width_ratio_between[1])
        crop_height = int(np.round(height_ratio * source_height))
        crop_width = int(np.round(width_ratio * source_width))
        height_leeway = source_height - crop_height
        width_leeway = source_width - crop_width
        # Set random variables
        if height_leeway > 0:
            self.set_random_variable('height_location',
                                     np.random.randint(low=0, high=height_leeway + 1))
        if width_leeway > 0:
            self.set_random_variable('width_location',
                                     np.random.randint(low=0, high=width_leeway + 1))
        self.set_random_variable('crop_height', crop_height)
        self.set_random_variable('crop_width', crop_width)
        self.set_random_variable('height_leeway', height_leeway)
        self.set_random_variable('width_leeway', width_leeway)

    def image_function(self, image):
        # Validate image shape
        if self._image_shape_cache is not None:
            assert_(self._image_shape_cache == image.shape,
                    "RandomCrop works on multiple images simultaneously only "
                    "if they have the same shape. Was expecting an image of "
                    "shape {}, got one of shape {} instead."
                    .format(self._image_shape_cache, image.shape),
                    ShapeError)
        else:
            self._image_shape_cache = image.shape
        height_leeway = self.get_random_variable('height_leeway', image_shape=image.shape)
        width_leeway = self.get_random_variable('width_leeway', image_shape=image.shape)
        if height_leeway > 0:
            height_location = self.get_random_variable('height_location',
                                                       image_shape=image.shape)
            crop_height = self.get_random_variable('crop_height',
                                                   image_shape=image.shape)
            cropped = image[height_location:(height_location + crop_height), :]
        else:
            cropped = image
        if width_leeway > 0:
            width_location = self.get_random_variable('width_location',
                                                      image_shape=image.shape)
            crop_width = self.get_random_variable('crop_width',
                                                  image_shape=image.shape)
            cropped = cropped[:, width_location:(width_location + crop_width)]
        return cropped


class RandomGammaCorrection(Transform):
    """Applies gamma correction [1] with a random gamma.

    This transform uses `skimage.exposure.adjust_gamma`, which requires the input be positive.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Gamma_correction
    """
    def __init__(self, gamma_between=(0.5, 2.), gain=1, **super_kwargs):
        """
        Parameters
        ----------
        gamma_between : tuple or list
            Specifies the range within which to sample gamma (uniformly).
        gain : int or float
            The resulting gamma corrected image is multiplied by this `gain`.
        super_kwargs : dict
            Keyword arguments for the superclass.
        """
        super(RandomGammaCorrection, self).__init__(**super_kwargs)
        self.gamma_between = list(gamma_between)
        self.gain = gain

    def build_random_variables(self):
        np.random.seed()
        self.set_random_variable('gamma',
                                 np.random.uniform(low=self.gamma_between[0],
                                                   high=self.gamma_between[1]))

    def image_function(self, image):
        gamma_adjusted = adjust_gamma(image,
                                      gamma=self.get_random_variable('gamma'),
                                      gain=self.gain)
        return gamma_adjusted


class ElasticTransform(Transform):
    """Random Elastic Transformation."""
    NATIVE_DTYPES = {'float32', 'float64'}
    PREFERRED_DTYPE = 'float32'

    def __init__(self, alpha, sigma, order=1, invert=False, **super_kwargs):
        self._initial_dtype = None
        super(ElasticTransform, self).__init__(**super_kwargs)
        self.alpha = alpha
        self.sigma = sigma
        self.order = order
        self.invert = invert

    def build_random_variables(self, **kwargs):
        # All this is done just once per batch (i.e. until `clear_random_variables` is called)
        np.random.seed()
        imshape = kwargs.get('imshape')
        # Build and scale random fields
        random_field_x = np.random.uniform(-1, 1, imshape) * self.alpha
        random_field_y = np.random.uniform(-1, 1, imshape) * self.alpha
        # Smooth random field (this has to be done just once per reset)
        sdx = gaussian_filter(random_field_x, self.sigma, mode='reflect')
        sdy = gaussian_filter(random_field_y, self.sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(imshape[1]), np.arange(imshape[0]))
        # Make inversion coefficient
        _inverter = 1. if not self.invert else -1.
        # Distort meshgrid indices (invert if required)
        flow_y, flow_x = (y + _inverter * sdy).reshape(-1, 1), (x + _inverter * sdx).reshape(-1, 1)
        # Set random states
        self.set_random_variable('flow_x', flow_x)
        self.set_random_variable('flow_y', flow_y)

    def cast(self, image):
        if image.dtype not in self.NATIVE_DTYPES:
            self._initial_dtype = image.dtype
            image = image.astype(self.PREFERRED_DTYPE)
        return image

    def uncast(self, image):
        if self._initial_dtype is not None:
            image = image.astype(self._initial_dtype)
        self._initial_dtype = None
        return image

    def image_function(self, image):
        # Cast image to one of the native dtypes (one which that is supported by scipy)
        image = self.cast(image)
        # Take measurements
        imshape = image.shape
        # Obtain flows
        flows = self.get_random_variable('flow_y', imshape=imshape), \
                self.get_random_variable('flow_x', imshape=imshape)
        # Map cooordinates from image to distorted index set
        transformed_image = map_coordinates(image, flows,
                                            mode='reflect', order=self.order).reshape(imshape)
        # Uncast image to the original dtype
        transformed_image = self.uncast(transformed_image)
        return transformed_image


class AdditiveGaussianNoise(Transform):
    """Add gaussian noise to the input."""
    def __init__(self, sigma, **super_kwargs):
        super(AdditiveGaussianNoise, self).__init__(**super_kwargs)
        self.sigma = sigma

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('noise', np.random.normal(loc=0, scale=self.sigma,
                                                           size=kwargs.get('imshape')))

    def image_function(self, image):
        image = image + self.get_random_variable('noise', imshape=image.shape)
        return image


class RandomRotate(Transform):
    """Random 90-degree rotations."""
    def __init__(self, **super_kwargs):
        super(RandomRotate, self).__init__(**super_kwargs)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('k', np.random.randint(0, 4))

    def image_function(self, image):
        return np.rot90(image, k=self.get_random_variable('k'))


class RandomTranspose(Transform):
    """Random 2d transpose."""
    def __init__(self, **super_kwargs):
        super(RandomTranspose, self).__init__(**super_kwargs)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('do_transpose', np.random.uniform() > 0.5)

    def image_function(self, image):
        if self.get_random_variable('do_transpose'):
            image = np.transpose(image)
        return image


class RandomFlip(Transform):
    """Random left-right or up-down flips."""
    def __init__(self, allow_lr_flips=True, allow_ud_flips=True, **super_kwargs):
        super(RandomFlip, self).__init__(**super_kwargs)
        self.allow_lr_flips = allow_lr_flips
        self.allow_ud_flips = allow_ud_flips

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('flip_lr', np.random.uniform() > 0.5)
        self.set_random_variable('flip_ud', np.random.uniform() > 0.5)

    def image_function(self, image):
        if self.allow_lr_flips and self.get_random_variable('flip_lr'):
            image = np.fliplr(image)
        if self.allow_ud_flips and self.get_random_variable('flip_ud'):
            image = np.flipud(image)
        return image


class CenterCrop(Transform):
    """ Crop patch of size `size` from the center of the image """
    def __init__(self, size, **super_kwargs):
        super(CenterCrop, self).__init__(**super_kwargs)
        assert isinstance(size, (int, tuple))
        self.size = (size, size) if isinstance(size, int) else size

    def image_function(self, image):
        h, w = image.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return image[x1:x1 + tw, y1:y1 + th]


class BinaryMorphology(Transform):
    """
    Apply a binary morphology operation on an image. Supported operations are dilation
    and erosion.
    """
    def __init__(self, mode, num_iterations=1, morphology_kwargs=None, **super_kwargs):
        """
        Parameters
        ----------
        mode : {'dilate', 'erode'}
            Whether to dilate or erode.
        num_iterations : int
            Number of iterations to apply the operation for.
        morphology_kwargs: dict
            Keyword arguments to the morphology function
            (i.e. `scipy.ndimage.morphology.binary_erosion` or
            `scipy.ndimage.morphology.binary_erosion`)
        super_kwargs : dict
            Keyword arguments to the superclass.
        """
        super(BinaryMorphology, self).__init__(**super_kwargs)
        # Validate and assign mode
        assert_(mode in ['dilate', 'erode'],
                "Mode must be one of ['dilate', 'erode']. Got {} instead.".format(mode),
                ValueError)
        self.mode = mode
        self.num_iterations = num_iterations
        self.morphology_kwargs = {} if morphology_kwargs is None else dict(morphology_kwargs)

    def image_function(self, image):
        if self.mode == 'dilate':
            transformed_image = binary_dilation(image, iterations=self.num_iterations,
                                                **self.morphology_kwargs)
        elif self.mode == 'erode':
            transformed_image = binary_erosion(image, iterations=self.num_iterations,
                                               **self.morphology_kwargs)
        else:
            raise ValueError
        # Cast transformed image to the right dtype and return
        return transformed_image.astype(image.dtype)


class BinaryDilation(BinaryMorphology):
    """Apply a binary dilation operation on an image."""
    def __init__(self, num_iterations=1, morphology_kwargs=None, **super_kwargs):
        super(BinaryDilation, self).__init__(mode='dilate', num_iterations=num_iterations,
                                             morphology_kwargs=morphology_kwargs,
                                             **super_kwargs)


class BinaryErosion(BinaryMorphology):
    """Apply a binary erosion operation on an image."""
    def __init__(self, num_iterations=1, morphology_kwargs=None, **super_kwargs):
        super(BinaryErosion, self).__init__(mode='erode', num_iterations=num_iterations,
                                            morphology_kwargs=morphology_kwargs,
                                            **super_kwargs)
