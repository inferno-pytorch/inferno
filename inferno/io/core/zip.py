from torch.utils.data.dataset import Dataset
from . import data_utils as du
from .base import SyncableDataset


class Zip(SyncableDataset):
    """
    Zip two or more datasets to one dataset. If the datasets implement synchronization primitives,
    they are all synchronized with the first dataset.
    """
    def __init__(self, *datasets, sync=True, transforms=None):
        super(Zip, self).__init__()
        assert len(datasets) >= 1
        assert all([isinstance(dataset, Dataset) for dataset in datasets])
        assert transforms is None or callable(transforms)
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
        fetched = [dataset[index] for dataset in self.datasets]
        if self.transforms is None:
            return fetched
        elif callable(self.transforms):
            return self.transforms(fetched)
        else:
            raise RuntimeError

    def __len__(self):
        if du.defines_base_sequence(self):
            return super(Zip, self).__len__()
        else:
            return min([len(dataset) for dataset in self.datasets])

    def __repr__(self):
        if len(self.datasets) > 3:
            return "Zip({}xDatasets)".format(len(self.datasets))
        else:
            return "Zip(" + ", ".join([dataset.__repr__() for dataset in self.datasets]) + ')'