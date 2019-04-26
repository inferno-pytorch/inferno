import numpy as np
import scipy
from scipy.ndimage import zoom
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from .base import Transform
from ...utils.exceptions import assert_

class RandomFlip3D(Transform):
    def __init__(self, **super_kwargs):
        super(RandomFlip3D, self).__init__(**super_kwargs)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('flip_lr', np.random.uniform() > 0.5)
        self.set_random_variable('flip_ud', np.random.uniform() > 0.5)
        self.set_random_variable('flip_z', np.random.uniform() > 0.5)

    def volume_function(self, volume):
        if self.get_random_variable('flip_lr'):
            volume = volume[:, :, ::-1]
        if self.get_random_variable('flip_ud'):
            volume = volume[:, ::-1, :]
        if self.get_random_variable('flip_z'):
            volume = volume[::-1, :, :]
        return volume


class RandomRot3D(Transform):
    def __init__(self, rot_range, p=0.125, reshape=False, order=0, **super_kwargs):
        super(RandomRot3D, self).__init__(**super_kwargs)
        self.rot_range = rot_range
        self.p = p
        self.reshape = reshape
        self.order = order

    def build_random_variables(self, **kwargs):
        np.random.seed()

        self.set_random_variable('do_z', np.random.uniform() < self.p)
        self.set_random_variable('do_y', np.random.uniform() < self.p)
        self.set_random_variable('do_x', np.random.uniform() < self.p)

        self.set_random_variable('angle_z', np.random.uniform(-self.rot_range, self.rot_range))
        self.set_random_variable('angle_y', np.random.uniform(-self.rot_range, self.rot_range))
        self.set_random_variable('angle_x', np.random.uniform(-self.rot_range, self.rot_range))

    def volume_function(self, volume):
        angle_z = self.get_random_variable('angle_z')
        angle_y = self.get_random_variable('angle_y')
        angle_x = self.get_random_variable('angle_x')

        # rotate along z-axis
        if self.get_random_variable('do_z'):
            volume = scipy.ndimage.interpolation.rotate(volume, angle_z,
                                                        order=self.order, mode='nearest',
                                                        axes=(0, 1), reshape=self.reshape)
        # rotate along y-axis
        if self.get_random_variable('do_y'):
            volume = scipy.ndimage.interpolation.rotate(volume, angle_y,
                                                        order=self.order, mode='nearest',
                                                        axes=(0, 2), reshape=self.reshape)
        # rotate along x-axis
        if self.get_random_variable('do_y'):
            volume = scipy.ndimage.interpolation.rotate(volume, angle_x,
                                                        order=self.order, mode='nearest',
                                                        axes=(1, 2), reshape=self.reshape)
        return volume


class AdditiveRandomNoise3D(Transform):
    """ Add gaussian noise to 3d volume

    Need to know input shape before application, but can be
    synchronized between different inputs (cf. `AdditiveNoise`)
    Arguments:
        shape: shape of input volumes
        std: standard deviation of gaussian
        super_kwargs: keyword arguments for `Transform` base class
    """
    def __init__(self, shape, std, **super_kwargs):
        super(AdditiveRandomNoise3D, self).__init__(**super_kwargs)
        self.shape = shape
        self.std = float(std)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('noise_vol',
                                 np.random.normal(loc=0.0, scale=self.std, size=self.shape))

    def volume_function(self, volume):
        noise_vol = self.get_random_variable('noise_vol')
        return volume + noise_vol


# TODO different options than gaussian
class AdditiveNoise(Transform):
    """ Add noise to 3d volume

    Do NOT need to know input shape before application, but CANNOT be
    synchronized between different inputs (cf. `AdditiveRandomNoise`)
    Arguments:
        sigma: sigma for noise
        mode: mode of distribution (only gaussian supported for now)
        super_kwargs: keyword arguments for `Transform` base class
    """
    def __init__(self, sigma, mode='gaussian', **super_kwargs):
        assert mode == 'gaussian'
        super().__init__(**super_kwargs)
        self.sigma = sigma

    # TODO check if volume is tensor and use torch functions in that case
    def volume_function(self, volume):
        volume += np.random.normal(loc=0, scale=self.sigma, size=volume.shape)
        return volume


class CentralSlice(Transform):
    def volume_function(self, volume):
        half_z = volume.shape[0] // 2
        return volume[half_z:half_z + 1, ...]


class VolumeCenterCrop(Transform):
    """ Crop patch of size `size` from the center of the volume """
    def __init__(self, size, **super_kwargs):
        super(VolumeCrop, self).__init__(**super_kwargs)
        assert isinstance(size, (int, tuple))
        self.size = (size, size, size) if isinstance(size, int) else size
        assert len(size) == 3

    def volume_function(self, volume):
        h, w, d = volume.shape
        th, tw, td = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        z1 = int(round((d - td) / 2.))
        return volume[x1:x1+tw, y1:y1+th, z1:z1+td]


class VolumeAsymmetricCrop(Transform):
    """ Crop `crop_left` from the left borders and `crop_right` from the right borders """
    def __init__(self, crop_left, crop_right, **super_kwargs):
        super(VolumeAsymmetricCrop, self).__init__(**super_kwargs)
        assert isinstance(crop_left, (list, tuple))
        assert isinstance(crop_right, (list, tuple))
        assert len(crop_left) == 3
        assert len(crop_right) == 3
        self.crop_left = crop_left
        self.crop_right = crop_right

    def volume_function(self, volume):
        x1, y1, z1 = self.crop_left
        x2, y2, z2 = (np.array(volume.shape) - np.array(self.crop_right)).astype('uint32')
        return volume[x1:x2, y1:y2, z1:z2]


class Slices2Channels(Transform):
    """ Needed for training 2D network with slices above/below as additional channels
        For the input data transforms one dimension (x, y or z) into channels
        For the target data just takes the central slice and discards all the rest"""
    def __init__(self, num_channels, downsampling = 1, **super_kwargs):
        super(Slices2Channels, self).__init__(**super_kwargs)
        self.channels = num_channels
        self.downsampling = downsampling
    def batch_function(self, batch):
        try:
            self.axis = batch[0].shape.index(self.channels)
        except ValueError:
            print ("The axis has the shape of the desired channels number!")
        half = int(self.channels/2)
        new_input = np.moveaxis(batch[0], self.axis, 0)
        #take every nth slice to the both directions of the central slice
        indices = []
        for i in range (self.channels):
            if i%self.downsampling == half%self.downsampling:
                indices.append(i)
        new_input = new_input[indices]   #num_chan after - int (num_chan/(2*downsample)) * 2 + 1
        new_target = np.moveaxis(batch[1], self.axis, 0)
        new_target = new_target[half]
        return (new_input, new_target)


class RandomScale3D(Transform):
    """Scales a volume with a random zoom factor with spline interpolation of requested order"""
    def __init__(self, zoom_factor_range, interpolation_order=0, p=0.5,
                 same_zoom=True, zoom_kwargs=None, **super_kwargs):
        """
        Parameters
        ----------
        zoom_factor_range : list or tuple
            The allowed range to sample zoom factors along the axes.
        interpolation_order : int
            Interpolation order for the spline interpolation.
        p : float
            Probability that the axis gets zoomed
        same_zoom: bool
            Apply the same zoom factor to all the axes
        zoom_kwargs : dict
            Keyword arguments for `scipy.ndimage.zoom`.
        super_kwargs : dict
            Keyword arguments for the superclass.
        """
        super(RandomScale3D, self).__init__(**super_kwargs)
        assert_(len(zoom_factor_range) == 2,
                    "`zoom_factor_range` must be a list or a tuple of length 2.",
                    ValueError)
        self.min = zoom_factor_range[0]
        self.max = zoom_factor_range[1]
        self.interpolation_order = interpolation_order
        self.p = p
        self.same_zoom = same_zoom
        self.zoom_kwargs = {} if zoom_kwargs is None else dict(zoom_kwargs)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('do_z', np.random.uniform() < self.p)
        self.set_random_variable('do_y', np.random.uniform() < self.p)
        self.set_random_variable('do_x', np.random.uniform() < self.p)
        self.set_random_variable('zoom_z', np.random.uniform(self.min, self.max))
        self.set_random_variable('zoom_y', np.random.uniform(self.min, self.max))
        self.set_random_variable('zoom_x', np.random.uniform(self.min, self.max))

    def volume_function(self, volume):
        zoom_z = self.get_random_variable('zoom_z') \
            if self.get_random_variable('do_z') else 1
        zoom_y = self.get_random_variable('zoom_y') \
            if self.get_random_variable('do_y') else 1
        zoom_x = self.get_random_variable('zoom_x') \
            if self.get_random_variable('do_x') else 1

        if self.same_zoom:
            zoom_y, zoom_x = zoom_z, zoom_z

        zoomed_volume = zoom(volume, (zoom_z, zoom_y, zoom_x),
                                 order=self.interpolation_order, **self.zoom_kwargs)
        return zoomed_volume


class RandomBinaryMorphology3D(Transform):
    """
    Apply a random binary morphology operation  (dilation or erosion).
    Allowed range of iteration number can be set.
    """
    def __init__(self, p=0.5, num_iter_range=(1,5), morphology_kwargs=None, **super_kwargs):
        """
        Parameters
        ----------
        p : float
            Probability that any operation is applied
        num_iter_range : list or tuple
            The allowed range of iteration number to apply the operation for.
        morphology_kwargs: dict
            Keyword arguments to the morphology function
            (i.e. `scipy.ndimage.morphology.binary_erosion` or
            `scipy.ndimage.morphology.binary_erosion`)
        super_kwargs : dict
            Keyword arguments to the superclass.
        """
        super(RandomBinaryMorphology3D, self).__init__(**super_kwargs)
        assert_(len(num_iter_range) == 2,
                    "`num_iter_range` must be a list or a tuple of length 2.",
                    ValueError)
        self.p = p
        self.min_iter = num_iter_range[0]
        self.max_iter = num_iter_range[1] + 1
        self.morphology_kwargs = {} if morphology_kwargs is None else dict(morphology_kwargs)

    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('do', np.random.uniform() < self.p)
        self.set_random_variable('erode', np.random.uniform() < 0.5)
        self.set_random_variable('iter_num', np.random.randint(self.min_iter, self.max_iter))

    def volume_function(self, volume):
        do = self.get_random_variable('do')
        erode_mode = self.get_random_variable('erode')
        iter_num = self.get_random_variable('iter_num')

        if do:
            if erode_mode:
                transformed_volume = binary_erosion(volume, iterations=iter_num,
                                                **self.morphology_kwargs)
            else:
                transformed_volume = binary_dilation(volume, iterations=iter_num,
                                                **self.morphology_kwargs)
            volume = transformed_volume.astype(volume.dtype)

        return volume


class CropPad2Divisible(Transform):
    """
    Given the number, symmetrically crops/pads the volume
    for all dimensions to be divisible by this number.
    Used e.g. to feed input with any shape to models with pooling layers.
    The threshold of cropping vs padding can be specified.
    """
    def __init__(self, divisor=16, crop_pad_threshold=0.2,
                 mode='constant', padding_kwargs=None, **super_kwargs):
        """
        Parameters
        ----------
        divisor : int
            A number that all dimensions should be divisible by
        crop_pad_threshold : float
            When "division remainder to divisor" ratio is lower then this number,
            input volume will be cropped, otherwise - padded.
            Set to 0 to only pad and 1 to only crop.
        mode: ‘constant’, ‘edge’, ‘symmetric’, etc
            See all the possible modes in numpy.pad doc
        padding_kwargs: dict
            Keyword arguments to numpy.pad
        super_kwargs : dict
            Keyword arguments to the superclass.
        """
        super(CropPad2Divisible, self).__init__(**super_kwargs)
        assert_(0 <= crop_pad_threshold <= 1,
                    "threshold must be between 0 and 1 inclusive",
                    ValueError)
        assert_(divisor%2==0, "divisor must be an even number", ValueError)
        self.divisor = divisor
        self.crop_pad_threshold = crop_pad_threshold
        self.mode = mode
        self.padding_kwargs = {} if padding_kwargs is None else dict(padding_kwargs)

    def volume_function(self, volume):
        half_div = int(self.divisor/2)
        remainders = [axis%self.divisor for axis in volume.shape]
        to_pad = [remainder/self.divisor >= self.crop_pad_threshold
               for remainder in remainders]
        diffs = [(int(np.floor(remainder/2)), int(np.ceil(remainder/2)))
                   for remainder in remainders]
        padding = [(half_div - diff[0], half_div - diff[1])
                    if pad else (0, 0)
                    for diff, pad in zip(diffs, to_pad)]
        cropping = [slice(diff[0], -diff[1])
                    if not (pad or diff[1]==0) else slice(None, None)
                    for diff, pad in zip(diffs, to_pad)]
        volume = np.pad(volume, pad_width=padding, mode=self.mode, **self.padding_kwargs)
        volume = volume[cropping]

        return volume
