from torch.utils.data.dataset import Dataset
import torch.multiprocessing as mp
import numpy as np
from . import data_utils as du
from .base import SyncableDataset
from ...utils.exceptions import assert_
from ...utils import python_utils as pyu
import random


class Zip(SyncableDataset):
    """
    Zip two or more datasets to one dataset. If the datasets implement synchronization primitives,
    they are all synchronized with the first dataset.
    """

    def __init__(self, *datasets, sync=False, transforms=None):
        super(Zip, self).__init__()
        assert_(len(datasets) >= 1, "Expecting one or more datasets, got none.", ValueError)
        for dataset_index, dataset in enumerate(datasets):
            assert_(isinstance(dataset, Dataset),
                    "Object at position {} of type {} is not a subclass of "
                    "`torch.utils.data.dataset.Dataset`"
                    .format(dataset_index, type(dataset).__name__),
                    TypeError)
        assert_(transforms is None or callable(transforms),
                "Given `transforms` is not callable.",
                TypeError)
        self.datasets = datasets
        self.sync = sync
        self.transforms = transforms
        if self.sync:
            self.sync_datasets()
        # Inherit base sequence if sync'ing
        if self.sync and all([du.defines_base_sequence(dataset) for dataset in self.datasets]):
            self.base_sequence = list(zip(*[dataset.base_sequence for dataset in self.datasets]))
        else:
            self.base_sequence = None

    def sync_datasets(self):
        master_dataset = self.datasets[0]
        for dataset in self.datasets[1:]:
            if du.implements_sync_primitives(dataset):
                dataset.sync_with(master_dataset)

    def sync_with(self, dataset):
        master_dataset = self.datasets[0]
        if du.implements_sync_primitives(master_dataset):
            master_dataset.sync_with(dataset)
        # Sync all other datasets
        self.sync_datasets()

    def __getitem__(self, index):
        assert_(index < len(self), exception_type=IndexError)
        fetched = [dataset[index] for dataset in self.datasets]
        if self.transforms is None:
            return fetched
        elif callable(self.transforms):
            return self.transforms(*fetched)
        else:
            raise RuntimeError

    def __len__(self):
        if du.defines_base_sequence(self):
            return super(Zip, self).__len__()
        else:
            return min([len(dataset) for dataset in self.datasets])

    def __repr__(self):
        if len(self.datasets) > 3:
            return "{}({}xDatasets)".format(type(self).__name__, len(self.datasets))
        else:
            return "{}(".format(type(self).__name__) + \
                   ", ".join([dataset.__repr__() for dataset in self.datasets[:-1]]) + ", " + \
                   self.datasets[-1].__repr__() + \
                   ')'


class ZipReject(Zip):
    """
    Extends `Zip` by the functionality of rejecting samples that don't fulfill
    a specified rejection criterion.
    """

    def __init__(self, *datasets, sync=False, transforms=None,
                 rejection_dataset_indices, rejection_criterion,
                 random_jump_after_reject=True):
        """
        Parameters
        ----------
        datasets : list or tuple
            Datasets to zip.
        sync : bool
            Whether to synchronize zipped datasets if a synchronization primitive is available.
        transforms : callable
            Transforms to apply on the fetched batch.
        rejection_dataset_indices : int or list or tuple
            Indices (or index) corresponding to the datasets which are used to determine whether
            a batch should be rejected.
        rejection_criterion : callable
            Criterion for rejection of batch. Must be a callable that accepts one or more
            arrays / tensors and returns True if the corresponding batch should be rejected,
            False otherwise. Should accept as many inputs as the number of elements in
            `rejection_dataset_indices` if the latter is a list, and 1 otherwise. Note that
            the order of the inputs to the `rejection_criterion` is the same as the order of
            the indices in `rejection_dataset_indices`.
        random_jump_after_reject: bool
            Whether to try a random index or the rejected index incremented by one after rejection.
        """
        super(ZipReject, self).__init__(*datasets, sync=sync, transforms=transforms)
        for rejection_dataset_index in pyu.to_iterable(rejection_dataset_indices):
            assert_(rejection_dataset_index < len(datasets),
                    "Index of the dataset to be used for rejection (= {}) is larger "
                    "than the number of datasets (= {}) minus one."
                    .format(rejection_dataset_index, len(datasets)),
                    IndexError)
        self.rejection_dataset_indices = pyu.to_iterable(rejection_dataset_indices)
        assert_(callable(rejection_criterion),
                "Rejection criterion is not callable as it should be.",
                TypeError)
        # return true if fetched should be rejected
        self.rejection_criterion = rejection_criterion
        # Array shared over processes to keep track of which indices have been rejected
        self.rejected = mp.Array('b', len(self))
        self.available_indices = None
        # optional index mapping to exclude rejected indices, reducing dataset size (see remove_rejected())
        self.index_mapping = None

        self.random_jump_after_reject = random_jump_after_reject

    def remove_rejected(self):
        # remove the indices belonging to samples that were rejected from the dataset
        # this changes the length of the dataset
        rejected = np.array(self.rejected[:])
        self.index_mapping = np.argwhere(1 - rejected)[:, 0]
        self.rejected = mp.Array('b', len(self))
        # just in case of num_workers == 0
        self.available_indices = None

    def __len__(self):
        if hasattr(self, 'index_mapping') and self.index_mapping is not None:
            return len(self.index_mapping)
        else:

            return super(ZipReject, self).__len__()

    def next_index_to_try(self, index):
        if self.random_jump_after_reject:
            return np.random.randint(len(self))
        else:
            return (index + 1) % len(self)

    def fetch_from_rejection_datasets(self, index):
        rejection_fetched = [self.datasets[rejection_dataset_index][index]
                             for rejection_dataset_index in self.rejection_dataset_indices]
        return rejection_fetched

    def __getitem__(self, index):
        # we increase the index until a valid batch of 'rejection_dataset' is found
        assert_(index < len(self), exception_type=IndexError)
        index_ = index
        # if we have a rejection dataset, check if the rejection criterion is fulfilled
        # and update the index
        if self.rejection_dataset_indices is not None:
            # at the start of each epoch, compute the available indices from the shared variable
            if self.available_indices is None:
                self.available_indices = set(np.argwhere(1 - np.array(self.rejected[:]))[:, 0])

            reject = True
            while reject:
                # check if there are no potentially valid indices left
                if not self.available_indices:
                    raise RuntimeError("ZipReject: No valid batch was found!")

                # check if this index was marked as rejected before
                if index_ not in self.available_indices:
                    index_ = self.next_index_to_try(index_)
                    continue
                # check if this index was marked as rejected in any process
                if self.rejected[index_]:
                    self.available_indices.remove(index_)
                    continue

                # map the index, if an index_mapping has been defined (see remove_rejected())
                mapped_index_ = index_ if self.index_mapping is None else self.index_mapping[index_]
                # we only fetch the dataset which has the rejection criterion
                # and only fetch all datasets when a valid index is found
                rejection_fetched = self.fetch_from_rejection_datasets(mapped_index_)
                # check if this batch is to be rejected
                reject = self.rejection_criterion(*rejection_fetched)

                # if so, increase the index and add it
                if reject:
                    self.rejected[index_] = True
                    self.available_indices.remove(index_)

            # fetch all other datasets and concatenate them with the valid rejection_fetch
            fetched = []
            for dataset_index, dataset in enumerate(self.datasets):
                if dataset_index in self.rejection_dataset_indices:
                    # Find the index in `rejection_fetched` corresponding to this dataset_index
                    index_in_rejection_fetched = self.rejection_dataset_indices.index(dataset_index)
                    # ... and append to fetched
                    fetched.append(rejection_fetched[index_in_rejection_fetched])
                else:
                    # Fetch and append to fetched
                    fetched.append(dataset[mapped_index_])
        else:
            # map the index, if an index_mapping has been defined (see remove_rejected())
            mapped_index_ = index_ if self.index_mapping is None else self.index_mapping[index_]
            fetched = [dataset[mapped_index_] for dataset in self.datasets]
        # apply transforms if present
        if self.transforms is not None:
            assert_(callable(self.transforms), "`self.transforms` is not callable.", TypeError)
            fetched = self.transforms(*fetched)
        return fetched
