import unittest
import os
import numpy as np

# try to load io libraries (h5py and z5py)
try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

# try:
#     import z5py
#     WITH_Z5PY = True
# except ImportError:
#     WITH_Z5PY = False


class TestLazyVolumeLoader(unittest.TestCase):

    def tearDown(self):
        try:
            os.remove('tmp.h5')
        except OSError:
            pass

    @unittest.skipUnless(WITH_H5PY, "Need h5py")
    def test_h5_loader(self):
        from inferno.io.volumetric.lazy_volume_loader import LazyHDF5VolumeLoader
        shape = (100, 100)

        # test default data loader
        data = np.arange(np.product(shape)).reshape(shape)
        with h5py.File('tmp.h5') as f:
            f.create_dataset('data', data=data)

        loader = LazyHDF5VolumeLoader('tmp.h5', 'data',
                                      window_size=[10, 10], stride=[10, 10],
                                      return_index_spec=True)
        self.assertEqual(loader.shape, shape)
        for batch, index in loader:
            expected = data[index.base_sequence_at_index]
            self.assertEqual(batch.shape, expected.shape)
            self.assertTrue(np.allclose(batch, expected))

    @unittest.skipUnless(WITH_H5PY, "Need h5py")
    def test_h5_loader_data_slice(self):
        from inferno.io.volumetric.lazy_volume_loader import LazyHDF5VolumeLoader
        shape = (100, 100, 100)
        data_slice = np.s_[:, 20:80, 10:30]

        # test default data loader
        data = np.arange(np.product(shape)).reshape(shape)
        with h5py.File('tmp.h5') as f:
            f.create_dataset('data', data=data)
        data = data[data_slice]

        loader = LazyHDF5VolumeLoader('tmp.h5', 'data',
                                      window_size=[10, 10, 10], stride=[10, 10, 10],
                                      return_index_spec=True, data_slice=data_slice)
        self.assertEqual(loader.shape, data.shape)
        for batch, index in loader:
            slice_ = index.base_sequence_at_index
            expected = data[slice_]
            self.assertEqual(batch.shape, expected.shape)
            self.assertTrue(np.allclose(batch, expected))

    @unittest.skipUnless(WITH_H5PY, "Need h5py")
    def test_h5_loader_pad(self):
        from inferno.io.volumetric.lazy_volume_loader import LazyHDF5VolumeLoader
        shape = (100, 100, 100)
        pad = [[0, 10], [0, 0], [5, 15]]

        # test default data loader
        data = np.arange(np.product(shape)).reshape(shape)
        with h5py.File('tmp.h5') as f:
            f.create_dataset('data', data=data)
        data = np.pad(data, pad_width=pad, mode='constant')

        loader = LazyHDF5VolumeLoader('tmp.h5', 'data',
                                      window_size=[20, 20, 20], stride=[20, 20, 20],
                                      return_index_spec=True, padding=pad, padding_mode='constant')
        self.assertEqual(loader.shape, data.shape)
        for batch, index in loader:
            slice_ = index.base_sequence_at_index
            expected = data[slice_]
            self.assertEqual(batch.shape, expected.shape)
            self.assertTrue(np.allclose(batch, expected))

    @unittest.skipUnless(WITH_H5PY, "Need h5py")
    def test_h5_loader_data_slice_pad(self):
        from inferno.io.volumetric.lazy_volume_loader import LazyHDF5VolumeLoader
        shape = (100, 100, 100)
        data_slice = np.s_[:, 20:80, 10:90]
        pad = [[0, 10], [5, 5], [5, 15]]

        # test default data loader
        data = np.arange(np.product(shape)).reshape(shape)
        with h5py.File('tmp.h5') as f:
            f.create_dataset('data', data=data)
        data = data[data_slice]
        data = np.pad(data, pad_width=pad, mode='constant')

        loader = LazyHDF5VolumeLoader('tmp.h5', 'data',
                                      window_size=[20, 20, 20], stride=[20, 20, 20],
                                      return_index_spec=True, padding=pad, padding_mode='constant',
                                      data_slice=data_slice)
        self.assertEqual(loader.shape, data.shape)
        for batch, index in loader:
            slice_ = index.base_sequence_at_index
            expected = data[slice_]
            self.assertEqual(batch.shape, expected.shape)
            self.assertTrue(np.allclose(batch, expected))


if __name__ == '__main__':
    unittest.main()
