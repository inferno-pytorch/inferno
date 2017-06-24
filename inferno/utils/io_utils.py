import os
import h5py as h5
import numpy as np


# Function to load in a dataset from a h5file
def fromh5(path, datapath=None, dataslice=None, asnumpy=True, preptrain=None):
    """
    Opens a hdf5 file at path, loads in the dataset at datapath, and returns dataset as a numpy array.
    :type path: str
    :param path: Path to h5 file

    :type datapath: str
    :param datapath: Path in h5 file (of the dataset). If not provided, returns the first dataset found.

    :type asnumpy: bool
    :param asnumpy: Whether to return as a numpy array (or a h5 dataset object)

    :type preptrain: prepkit.preptrain
    :param preptrain: Train of preprocessing functions to be applied on the dataset before being returned
    """
    # Check if path exists (thanks Lukas!)
    assert os.path.exists(path), "Path {} does not exist.".format(path)
    # Init file
    h5file = h5.File(path)
    # Init dataset
    h5dataset = h5file[datapath] if datapath is not None else h5file.values()[0]
    # Slice dataset
    h5dataset = h5dataset[dataslice] if dataslice is not None else h5dataset
    # Convert to numpy if required
    h5dataset = np.asarray(h5dataset) if asnumpy else h5dataset
    # Apply preptrain
    h5dataset = preptrain(h5dataset) if preptrain is not None else h5dataset
    # Close file
    h5file.close()
    # Return
    return h5dataset


def toh5(data, path, datapath='data'):
    """
    Write `data` to a HDF5 volume.

    :type data: numpy.ndarray
    :param data: Data to write.

    :type path: str
    :param path: Path to the volume.

    :type datapath: str
    :param datapath: Path to the volume in the HDF5 volume.
    """
    with h5.File(path, 'w') as f:
        f.create_dataset(datapath, data=data)
