import numpy as np
from .base import Transform
import numpy as np
from .base import Transform

import scipy




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
    def __init__(self, rot_range, p=0.125,  only_one=True, **super_kwargs):
        super(RandomRot3D, self).__init__(**super_kwargs)
        self.rot_range = rot_range
        self.p = p
    def build_random_variables(self, **kwargs):
        np.random.seed()

        self.set_random_variable('do_z',  np.random.uniform() < self.p)
        self.set_random_variable('do_y',  np.random.uniform() < self.p)
        self.set_random_variable('do_x',  np.random.uniform() < self.p)


        self.set_random_variable('angle_z',  np.random.uniform(-self.rot_range, self.rot_range) )
        self.set_random_variable('angle_y',  np.random.uniform(-self.rot_range, self.rot_range) )
        self.set_random_variable('angle_x',   np.random.uniform(-self.rot_range, self.rot_range) )

    def volume_function(self, volume):


        #print("inshape", volume.shape)
        angle_z = self.get_random_variable('angle_z')
        angle_y = self.get_random_variable('angle_y')
        angle_x = self.get_random_variable('angle_x')
        

        # rotate along z-axis
        if self.get_random_variable('do_z'):
            volume = scipy.ndimage.interpolation.rotate(volume, angle_z, order=0, mode='nearest', axes=(0, 1), reshape=False)
            #return volume
        # rotate along y-axis
        if self.get_random_variable('do_y'):
            volume = scipy.ndimage.interpolation.rotate(volume, angle_y, order=0, mode='nearest', axes=(0, 2), reshape=False)
            #return volume
        # rotate along x-axis
        if self.get_random_variable('do_y'):
            volume = scipy.ndimage.interpolation.rotate(volume, angle_x, order=0, mode='nearest', axes=(1, 2), reshape=False)
            #return volume

        return volume



class AdditiveRandomNoise3D(Transform):
    def __init__(self, shape, std, **super_kwargs):
        super(AdditiveRandomNoise3D, self).__init__(**super_kwargs)
        self.shape = shape
        self.std = float(std)
    def build_random_variables(self, **kwargs):
        np.random.seed()
        self.set_random_variable('noise_vol',  np.random.normal(loc=0.0, scale=self.std, size=self.shape) )


    def volume_function(self, volume):


        #print("inshape", volume.shape)
        noise_vol = self.get_random_variable('noise_vol')

        return volume + noise_vol


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
