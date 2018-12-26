import unittest
import os
from shutil import rmtree

import numpy as np
import h5py


class TestVolumeLoader(unittest.TestCase):
    shape = (100, 100, 100)
    def setUp(self):
        self.data = np.random.rand(*self.shape)

    def test_loader(self):
        from inferno.io.volumetric import VolumeLoader
        loader = VolumeLoader(self.data,
                              window_size=(10, 10, 10),
                              stride=(10, 10, 10), return_index_spec=True)
        for batch, idx in loader:
            slice_ = loader.base_sequence[int(idx)]
            expected = self.data[slice_]
            self.assertEqual(batch.shape, expected.shape)
            self.assertTrue(np.allclose(batch, expected))


class TestHDF5VolumeLoader(unittest.TestCase):
    shape = (100, 100, 100)
    def setUp(self):
        try:
            os.mkdir('./tmp')
        except OSError:
            pass
        self.data = np.random.rand(*self.shape)
        with h5py.File('./tmp/data.h5') as f:
            f.create_dataset('data', data=self.data)

    def tearDown(self):
        try:
            rmtree('./tmp')
        except OSError:
            pass

    def test_hdf5_loader(self):
        from inferno.io.volumetric import HDF5VolumeLoader
        loader = HDF5VolumeLoader('./tmp/data.h5', 'data',
                                  window_size=(10, 10, 10),
                                  stride=(10, 10, 10), return_index_spec=True)
        for batch, idx in loader:
            slice_ = loader.base_sequence[int(idx)]
            expected = self.data[slice_]
            self.assertEqual(batch.shape, expected.shape)
            self.assertTrue(np.allclose(batch, expected))



if __name__ == '__main__':
    unittest.main()
