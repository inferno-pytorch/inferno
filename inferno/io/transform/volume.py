import numpy as np
from .base import Transform


class RandomFlip3D(Transform):
    def __init__(self, rng=np.random.RandomState(42), **super_kwargs):
        super(RandomFlip3D, self).__init__(**super_kwargs)
        self.rng = rng

    def build_random_variables(self, **kwargs):
        self.set_random_variable('flip_lr', self.rng.uniform() > 0.5)
        self.set_random_variable('flip_ud', self.rng.uniform() > 0.5)
        self.set_random_variable('flip_z', self.rng.uniform() > 0.5)

    def volume_function(self, volume):
        if self.get_random_variable('flip_lr'):
            volume = volume[:, :, ::-1]
        if self.get_random_variable('flip_ud'):
            volume = volume[:, ::-1, :]
        if self.get_random_variable('flip_z'):
            volume = volume[::-1, :, :]
        return volume


class CentralSlice(Transform):
    def volume_function(self, volume):
        half_z = volume.shape[0] // 2
        return volume[half_z:half_z + 1, ...]
