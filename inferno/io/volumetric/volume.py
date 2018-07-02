import numpy as np
import os
import skimage.io

# try to load io libraries (h5py and z5py)
try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

try:
    import z5py
    WITH_Z5PY = True
except ImportError:
    WITH_Z5PY = False

from ..core.base import SyncableDataset
from ..core.base import IndexSpec
from . import volumetric_utils as vu
from ...utils import python_utils as pyu


class VolumeLoader(SyncableDataset):
    def __init__(self, volume, window_size, stride, downsampling_ratio=None, padding=None,
                 padding_mode='reflect', transforms=None, return_index_spec=False, name=None):
        super(VolumeLoader, self).__init__()
        # Validate volume
        assert isinstance(volume, np.ndarray)
        # Validate window size and stride
        assert len(window_size) == volume.ndim
        assert len(stride) == volume.ndim
        # Validate transforms
        assert transforms is None or callable(transforms)

        self.name = name
        self.return_index_spec = return_index_spec
        self.volume = volume
        self.window_size = window_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.transforms = transforms
        # DataloaderIter should do the shuffling
        self.shuffle = False

        if downsampling_ratio is None:
            self.downsampling_ratio = [1] * self.volume.ndim
        elif isinstance(downsampling_ratio, int):
            self.downsampling_ratio = [downsampling_ratio] * self.volume.ndim
        elif isinstance(downsampling_ratio, (list, tuple)):
            assert len(downsampling_ratio) == self.volume.ndim
            self.downsampling_ratio = list(downsampling_ratio)
        else:
            raise NotImplementedError

        if padding is None:
            self.padding = [[0, 0]] * self.volume.ndim
        else:
            self.padding = padding
            self.pad_volume()

        self.base_sequence = self.make_sliding_windows()

    # TODO this only works for numpy data, but we wanna support
    # h5py-like datasets as well. The easiest solution for this
    # would be to do the padding in `__getitem__` for this case
    def pad_volume(self, padding=None):
        padding = self.padding if padding is None else padding
        if padding is None:
            return self.volume
        else:
            self.volume = np.pad(self.volume,
                                 pad_width=self.padding,
                                 mode=self.padding_mode)
            return self.volume

    def make_sliding_windows(self):
        return list(vu.slidingwindowslices(shape=list(self.volume.shape),
                                           window_size=self.window_size,
                                           strides=self.stride,
                                           shuffle=self.shuffle,
                                           add_overhanging=True))

    # TODO pad on the fly for datasets here ?
    # TODO offset and check bounds for `data_slice is not None` here ?
    def __getitem__(self, index):
        # Casting to int would allow index to be IndexSpec objects.
        index = int(index)
        slices = self.base_sequence[index]
        sliced_volume = self.volume[tuple(slices)]
        if self.transforms is None:
            transformed = sliced_volume
        else:
            transformed = self.transforms(sliced_volume)
        if self.return_index_spec:
            return transformed, IndexSpec(index=index, base_sequence_at_index=slices)
        else:
            return transformed

    def clone(self, volume=None, transforms=None, name=None):
        # Make sure the volume shapes check out
        assert volume.shape == self.volume.shape
        # Make a new instance (without initializing)
        new = type(self).__new__(type(self))
        # Update dictionary to initialize
        new_dict = dict(self.__dict__)
        if volume is not None:
            new_dict.update({'volume': volume})
        if transforms is not None:
            new_dict.update({'transforms': transforms})
        if name is not None:
            new_dict.update({'name': name})
        new.__dict__.update(new_dict)
        return new

    def __repr__(self):
        return "{}(shape={}, name={})".format(type(self).__name__, self.volume.shape, self.name)


# TODO this should implement contextmanager
# baseclass for hdf5, zarr or n5 volume loaders
class ChunkedVolumeLoader(VolumeLoader):
    def __init__(self, file_impl, path,
                 path_in_file=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):

        if isinstance(path, dict):
            assert name is not None
            assert name in path
            self.path = path.get(name)
        elif isinstance(path, str):
            assert os.path.exists(path)
            self.path = path
        else:
            raise NotImplementedError

        if isinstance(path_in_file, dict):
            assert name is not None
            assert name in path_in_file
            self.path_in_file = path_in_file.get(name)
        elif isinstance(path_in_file, str):
            self.path_in_file = path_in_file
        elif path_in_file is None:
            self.path_in_file = None
        else:
            raise NotImplementedError

        if data_slice is None or isinstance(data_slice, (str, list)):
            self.data_slice = vu.parse_data_slice(data_slice)
        elif isinstance(data_slice, dict):
            assert name is not None
            assert name in data_slice
            self.data_slice = vu.parse_data_slice(data_slice.get(name))
        else:
            raise NotImplementedError

        slicing_config_for_name = pyu.get_config_for_name(slicing_config, name)

        assert 'window_size' in slicing_config_for_name
        assert 'stride' in slicing_config_for_name

        # Read in volume from file
        # TODO we don't read the whole file immediately, but just open the corresponding dataset
        # however, with this implementing `data_slice` is not that trivial
        # an option would be to move `data_slice` to `VolumeLoader` and then change its
        # `__getitem__` accordingly; and also change the shape used for the sliding-window
        assert data_slice is None
        # volume = iou.fromh5(self.path, self.path_in_h5_dataset,
        #                     dataslice=(tuple(self.data_slice)
        #                                if self.data_slice is not None
        #                                else None))
        # we need to close this for h5 later
        self.file = file_impl(self.path)
        # Initialize superclass with the volume
        super(ChunkedVolumeLoader, self).__init__(volume=self.file[self.path_in_file], name=name, transforms=transforms,
                                                  **slicing_config_for_name)


class HDF5VolumeLoader(ChunkedVolumeLoader):
    # TODO the name `path_in_h5_dataset` does not make sense.
    # should be `path_in_file` (or `path_in_h5_file`); change or keep for legacy?
    def __init__(self, path, path_in_h5_dataset=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):
        assert WITH_H5PY, "Need h5py to load volume from hdf5 file."
        super(HDF5VolumeLoader, self).__init__(file_impl=h5py.File, path=path,
                                               path_in_file=path_in_h5_dataset,
                                               data_slice=data_slice, transforms=transforms,
                                               name=name, **slicing_config)

    # this is not pythonic, but we need to close the h5py file
    def __del__(self):
        self.file.close()


class N5VolumeLoader(ChunkedVolumeLoader):
    def __init__(self, path, path_in_file=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):
        assert WITH_Z5PY, "Need z5py to load volume from N5 file."
        super(N5VolumeLoader, self).__init__(file_impl=z5py.N5File, path=path,
                                             path_in_file=path_in_file,
                                             data_slice=data_slice, transforms=transforms,
                                             name=name, **slicing_config)


class ZarrVolumeLoader(ChunkedVolumeLoader):
    def __init__(self, path, path_in_file=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):
        assert WITH_Z5PY, "Need z5py to load volume from zarr file."
        super(ZarrVolumeLoader, self).__init__(file_impl=z5py.ZarrFile, path=path,
                                               path_in_file=path_in_file,
                                               data_slice=data_slice, transforms=transforms,
                                               name=name, **slicing_config)


class TIFVolumeLoader(VolumeLoader):
    """Loader for volumes stored in .tif files."""
    def __init__(self, path, data_slice=None, transforms=None, name=None, **slicing_config):
        """
        Parameters
        ----------
        path : str
            Path to the volume.
        transforms : callable
            Transforms to apply on the read volume.
        slicing_config : dict
            Dictionary specifying the sliding window. Must contain keys 'window_size'
            and 'stride'.
        """
        if isinstance(path, dict):
            assert name in path.keys()
            assert os.path.exists(path.get(name))
            self.path = path.get(name)
        elif isinstance(path, str):
            assert os.path.exists(path)
            self.path = path
        else:
            raise NotImplementedError

        assert 'window_size' in slicing_config
        assert 'stride' in slicing_config

        if data_slice is None or isinstance(data_slice, (str, list)):
            self.data_slice = vu.parse_data_slice(data_slice)
        elif isinstance(data_slice, dict):
            assert name is not None
            assert name in data_slice
            self.data_slice = vu.parse_data_slice(data_slice.get(name))
        else:
            raise NotImplementedError

        # Read in volume from file
        volume = skimage.io.imread(self.path)
        # and slice it
        volume = volume[self.data_slice] if self.data_slice is not None else volume
        # Initialize superclass with the volume
        super(TIFVolumeLoader, self).__init__(volume=volume, transforms=transforms,
                                              **slicing_config)
