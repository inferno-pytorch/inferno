import numpy as np
import os

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


class LazyVolumeLoaderBase(SyncableDataset):
    def __init__(self, dataset, window_size, stride, downsampling_ratio=None, padding=None,
                 padding_mode='reflect', transforms=None, return_index_spec=False, name=None,
                 data_slice=None):
        super(LazyVolumeLoaderBase, self).__init__()
        assert len(window_size) == dataset.ndim, "%i, %i" % (len(window_size), dataset.ndim)
        assert len(stride) == dataset.ndim
        # Validate transforms
        assert transforms is None or callable(transforms)

        self.name = name
        self.return_index_spec = return_index_spec
        self.dataset = dataset
        self.window_size = window_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.transforms = transforms
        # slicing and padding
        self.data_slice = self.normalize_slice(data_slice)
        self.padding = padding
        # DataloaderIter should do the shuffling
        self.shuffle = False

        # compute the shape
        self.shape = self.get_shape()
        self._data_shape = tuple(dsl.stop - dsl.start for dsl in self.data_slice)\
            if self.data_slice is not None else self.dataset.shape

        if downsampling_ratio is None:
            self.downsampling_ratio = [1] * self.dataset.ndim
        elif isinstance(downsampling_ratio, int):
            self.downsampling_ratio = [downsampling_ratio] * self.dataset.ndim
        elif isinstance(downsampling_ratio, (list, tuple)):
            assert len(downsampling_ratio) == self.dataset.ndim
            self.downsampling_ratio = list(downsampling_ratio)
        else:
            raise NotImplementedError

        self.base_sequence = self.make_sliding_windows()

    def normalize_slice(self, data_slice):
        if data_slice is None:
            return None
        slice_ = tuple(slice(0 if sl.start is None else sl.start,
                             sh if sl.stop is None else sl.stop)
                       for sl, sh in zip(data_slice, self.dataset.shape))
        if len(slice_) < self.dataset.ndim:
            slice_ = slice_ + tuple(slice(0, sh) for sh in self.dataset.shape[len(slice_):])
        return slice_

    # get the effective shape after slicing and / or padding
    def get_shape(self):
        if self.data_slice is None:
            shape = self.dataset.shape
        else:
            # get the shape from the data slice (don't support ellipses)
            shape = tuple(slice_.stop - slice_.start for slice_ in self.data_slice)
        if self.padding is not None:
            # TODO is this correct ???
            shape = tuple(sh + sum(pad) for sh, pad in zip(shape, self.padding))
        return shape

    def make_sliding_windows(self):
        return list(vu.slidingwindowslices(shape=list(self.shape),
                                           window_size=self.window_size,
                                           strides=self.stride,
                                           shuffle=self.shuffle,
                                           add_overhanging=True,
                                           ds=self.downsampling_ratio))

    def __getitem__(self, index):
        # Casting to int would allow index to be IndexSpec objects.
        index = int(index)
        slices = self.base_sequence[index]

        slices_ = tuple(slices)

        # check if we have padding and if we need to pad
        if self.padding is not None:

            # get the start and stop positions in the dataset without padding
            starts = [sl.start - pad[0] for sl, pad in zip(slices_, self.padding)]
            stops = [sl.stop - pad[0] for sl, pad in zip(slices_, self.padding)]

            # check if we need to pad to the left
            pad_left = None
            if any(start < 0 for start in starts):
                pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
                starts = [max(0, start) for start in starts]

            # check if we need to pad to the right
            pad_right = None
            if any(stop > sh for stop, sh in zip(stops, self._data_shape)):
                pad_right = tuple(stop - sh if stop > sh else 0
                                  for stop, sh in zip(stops, self._data_shape))
                stops = [min(sh, stop) for sh, stop in zip(self._data_shape, stops)]

            # check if we need any paddingand if so calculate the padding width
            need_padding = pad_left is not None or pad_right is not None
            if need_padding:
                # check the pad width (left and right) that we need for this batch
                pad_left = (0,) * len(self.shape) if pad_left is None else pad_left
                pad_right = (0,) * len(self.shape) if pad_right is None else pad_right
                pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))

            # update the slicing
            slices_ = tuple(slice(start, stop) for start, stop in zip(starts, stops))
        else:
            need_padding = False

        # if we have data-slices, we need to bring
        # the slices back to the volume space
        if self.data_slice is not None:
            slices_ = tuple(slice(sl.start + dsl.start, sl.stop + dsl.start)
                            for sl, dsl in zip(slices_, self.data_slice))

        # load the slice and pad if necessary
        sliced_volume = self.dataset[slices_]
        if need_padding:
            sliced_volume = np.pad(sliced_volume, pad_width=pad_width,
                                   mode=self.padding_mode)

        if self.transforms is None:
            transformed = sliced_volume
        else:
            transformed = self.transforms(sliced_volume)
        if self.return_index_spec:
            return transformed, IndexSpec(index=index, base_sequence_at_index=slices)
        else:
            return transformed

    def clone(self, dataset=None, transforms=None, name=None):
        # Make sure the dataset shapes check out
        assert dataset.shape == self.dataset.shape
        # Make a new instance (without initializing)
        new = type(self).__new__(type(self))
        # Update dictionary to initialize
        new_dict = dict(self.__dict__)
        if dataset is not None:
            new_dict.update({'dataset': dataset})
        if transforms is not None:
            new_dict.update({'transforms': transforms})
        if name is not None:
            new_dict.update({'name': name})
        new.__dict__.update(new_dict)
        return new

    def __repr__(self):
        return "{}(shape={}, name={})".format(type(self).__name__, self.dataset.shape, self.name)


# baseclass for hdf5, zarr or n5 volume loaders
class LazyVolumeLoader(LazyVolumeLoaderBase):
    def __init__(self, file_impl, path,
                 path_in_file=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):

        if isinstance(path, dict):
            assert name is not None
            assert name in path
            self.path = path.get(name)
        elif isinstance(path, str):
            assert os.path.exists(path), path
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

        if data_slice is None or isinstance(data_slice, (str, list, tuple)):
            data_slice = vu.parse_data_slice(data_slice)
        elif isinstance(data_slice, dict):
            assert name is not None
            assert name in data_slice
            data_slice = vu.parse_data_slice(data_slice.get(name))
        else:
            raise NotImplementedError
        self.validate_data_slice(data_slice)

        slicing_config_for_name = pyu.get_config_for_name(slicing_config, name)

        assert 'window_size' in slicing_config_for_name
        assert 'stride' in slicing_config_for_name

        self.file_ = file_impl(self.path, mode='r')
        # Initialize superclass with the volume
        super(LazyVolumeLoader, self).__init__(dataset=self.file_[self.path_in_file], name=name,
                                               transforms=transforms, data_slice=data_slice,
                                               **slicing_config_for_name)

    # we do not support step in the dataslice
    def validate_data_slice(self, data_slice):
        if data_slice is not None:
            assert all(sl.step in (None, 1) for sl in data_slice), "Complicated step is not supported"


class LazyHDF5VolumeLoader(LazyVolumeLoader):
    def __init__(self, path, path_in_h5_dataset=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):
        assert WITH_H5PY, "Need h5py to load volume from hdf5 file."
        super(LazyHDF5VolumeLoader, self).__init__(file_impl=h5py.File, path=path,
                                                   path_in_file=path_in_h5_dataset,
                                                   data_slice=data_slice, transforms=transforms,
                                                   name=name, **slicing_config)

    # this is not pythonic, but we need to close the h5py file
    def __del__(self):
        self.file_.close()


class LazyN5VolumeLoader(LazyVolumeLoader):
    def __init__(self, path, path_in_file=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):
        assert WITH_Z5PY, "Need z5py to load volume from N5 file."
        assert slicing_config.get('downsampling_ratio', None) is None,\
            "Downsampling is not supported by z5py based loaderes"
        super(N5VolumeLoader, self).__init__(file_impl=z5py.N5File, path=path,
                                             path_in_file=path_in_file,
                                             data_slice=data_slice, transforms=transforms,
                                             name=name, **slicing_config)


class LazyZarrVolumeLoader(LazyVolumeLoader):
    def __init__(self, path, path_in_file=None, data_slice=None, transforms=None,
                 name=None, **slicing_config):
        assert WITH_Z5PY, "Need z5py to load volume from zarr file."
        assert slicing_config.get('downsampling_ratio', None) is None,\
            "Downsampling is not supported by z5py based loaderes"
        super(ZarrVolumeLoader, self).__init__(file_impl=z5py.ZarrFile, path=path,
                                               path_in_file=path_in_file,
                                               data_slice=data_slice, transforms=transforms,
                                               name=name, **slicing_config)
