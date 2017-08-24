from torch.utils.data.dataset import Dataset
from . import data_utils as du
from .base import SyncableDataset
from ...utils.exceptions import assert_
from ...utils import python_utils as pyu


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
                 rejection_dataset_indices, rejection_criterion):
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
        self.rejection_criterion = rejection_criterion  # return true if fetched should be rejected

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
            # we only fetch the dataset which has the rejection criterion
            # and only fetch all datasets when a valid index is found
            rejection_fetched = self.fetch_from_rejection_datasets(index_)
            num_fetch_attempts = 0
            while self.rejection_criterion(*rejection_fetched):
                index_ = (index_ + 1) % len(self)
                rejection_fetched = self.fetch_from_rejection_datasets(index_)
                num_fetch_attempts += 1
                if num_fetch_attempts >= len(self):
                    raise RuntimeError("ZipReject: No valid batch was found!")
            # fetch all other datasets and concatenate them with the valid rejection_fetch
            fetched = []
            for dataset_index, dataset in enumerate(self.datasets):
                if dataset_index in self.rejection_dataset_indices:
                    # Find the index in `rejection_fetched` corresponding to this dataset_index
                    index_in_rejection_fetched = \
                        self.rejection_dataset_indices.index(dataset_index)
                    # ... and append to fetched
                    fetched.append(rejection_fetched[index_in_rejection_fetched])
                else:
                    # Fetch and append to fetched
                    fetched.append(dataset[index_])
        else:
            fetched = [dataset[index_] for dataset in self.datasets]
        # apply transforms if present
        if self.transforms is not None:
            assert_(callable(self.transforms), "`self.transforms` is not callable.", TypeError)
            fetched = self.transforms(*fetched)
        return fetched

