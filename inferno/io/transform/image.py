import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_dilation, binary_erosion

from .base import Transform
from ...utils.exceptions import assert_


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


class RandomFlip(Transform):
    """Random left-right or up-down flips."""
    def __init__(self, **super_kwargs):
        super(RandomFlip, self).__init__(**super_kwargs)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('flip_lr', np.random.uniform() > 0.5)
        self.set_random_variable('flip_ud', np.random.uniform() > 0.5)

    def image_function(self, image):
        if self.get_random_variable('flip_lr'):
            image = np.fliplr(image)
        if self.get_random_variable('flip_ud'):
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
        return image[x1:x1+tw, y1:y1+th]


class RandomCrop(Transform):
    """ Crop patch of size `size` from a random position of the image """
    def __init__(self, size, **super_kwargs):
        super(RandomCrop, self).__init__(**super_kwargs)
        assert isinstance(size, (int, tuple))
        self.size = (size, size) if isinstance(size, int) else size

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('x0', np.random.uniform())
        self.set_random_variable('y0', np.random.uniform())

    def image_function(self, image):
        h, w = image.shape
        th, tw = self.size

        safe_ymax = h - th
        safe_xmax = w - tw

        x0 = int(round(safe_xmax * self.get_random_variable('x0')))
        y0 = int(round(safe_ymax * self.get_random_variable('y0')))

        return image[x0:x0+tw, y0:y0+th]


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
