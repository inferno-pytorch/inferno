import os
import h5py as h5
import numpy as np
import yaml
from skimage.io import imsave


# Function to load in a dataset from a h5file
def fromh5(path, datapath=None, dataslice=None, asnumpy=True, preptrain=None):
    """
    Opens a hdf5 file at path, loads in the dataset at datapath, and returns dataset
    as a numpy array.
    """
    # Check if path exists (thanks Lukas!)
    assert os.path.exists(path), "Path {} does not exist.".format(path)
    with h5.File(path, 'r') as f:
        # Init dataset
        h5dataset = f[datapath] if datapath is not None else f.values()[0]
        # Slice dataset
        h5dataset = h5dataset[dataslice] if dataslice is not None else h5dataset
        # Convert to numpy if required
        h5dataset = np.asarray(h5dataset) if asnumpy else h5dataset
        # Apply preptrain
        h5dataset = preptrain(h5dataset) if preptrain is not None else h5dataset
    return h5dataset


# TODO we could also do **h5_kwargs instead
def toh5(data, path, datapath='data', compression=None, chunks=None):
    """Write `data` to a HDF5 volume."""
    with h5.File(path) as f:
        f.create_dataset(datapath, data=data, compression=compression, chunks=chunks)


def fromz5(path, datapath, dataslice=None, n_threads=8):
    # we import z5py only here because we don't want to assume that it's in the env
    import z5py
    assert os.path.exists(path), "Path {} does not exist.".format(path)
    with z5py.File(path) as f:
        ds = f[datapath]
        ds.n_threads = n_threads
        data = ds[:] if dataslice is None else ds[dataslice]
    return data


# Yaml to dict reader
def yaml2dict(path):
    if isinstance(path, dict):
        # Forgivable mistake that path is a dict already
        return path
    with open(path, 'r') as f:
        readict = yaml.load(f)
    return readict


def print_tensor(tensor, prefix, directory):
    """Prints a image or volume tensor to file as images."""
    def _print_image(image, prefix, batch, channel, z=None):
        if z is None:
            file_name = "{}--B-{}--CH-{}.png".format(prefix, batch, channel)
        else:
            file_name = "{}--B-{}--CH-{}--Z-{}.png".format(prefix, batch, channel, z)
        full_file_name = os.path.join(directory, file_name)
        imsave(arr=image, fname=full_file_name)

    for batch in range(tensor.shape[0]):
        for channel in range(tensor.shape[1]):
            if tensor.ndim == 4:
                _print_image(tensor[batch, channel, ...], prefix, batch, channel)
            else:
                for plane in range(tensor.shape[2]):
                    _print_image(tensor[batch, channel, plane, ...], prefix, batch, channel, plane)
