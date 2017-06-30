import numpy as np
from torch.utils.data.dataset import Dataset
from ...utils import python_utils as pyu


class Concatenate(Dataset):
    """
    Concatenates mutliple datasets to one. This class does not implement
    synchronization primitives.
    """
    def __init__(self, *datasets, transforms=None):
        assert all([isinstance(dataset, Dataset) for dataset in datasets])
        assert len(datasets) >= 1
        assert transforms is None or callable(transforms)
        self.datasets = datasets
        self.transforms = transforms

    def map_index(self, index):
        # Get a list of lengths of all datasets. Say the answer is [4, 3, 3],
        # and we're looking for index = 5.
        len_list = list(map(len, self.datasets))
        # Cumulate to a numpy array. The answer is [4, 7, 10]
        cumulative_len_list = np.cumsum(len_list)
        # When the index is subtracted, we get [-1, 2, 5]. We're looking for the (index
        # of the) first cumulated len which is larger than the index (in this case,
        # 7 (index 1)).
        offset_cumulative_len_list = cumulative_len_list - index
        dataset_index = np.argmax(offset_cumulative_len_list > 0)
        # With the dataset index, we figure out the index in dataset
        if dataset_index == 0:
            # First dataset - index corresponds to index_in_dataset
            index_in_dataset = index
        else:
            # Get cumulated length up to the current dataset
            len_up_to_dataset = cumulative_len_list[dataset_index - 1]
            # Compute index_in_dataset as that what's left
            index_in_dataset = index - len_up_to_dataset
        return dataset_index, index_in_dataset

    def __getitem__(self, index):
        assert index < len(self)
        dataset_index, index_in_dataset = self.map_index(index)
        fetched = self.datasets[dataset_index][index_in_dataset]
        if self.transforms is None:
            return fetched
        elif callable(self.transforms):
            return self.transforms(*pyu.to_iterable(fetched))
        else:
            raise NotImplementedError

    def __len__(self):
        return sum([len(dataset) for dataset in self.datasets])

    def __repr__(self):
        if len(self.datasets) < 3:
            return "Concatenate(" + \
                   ", ".join([dataset.__repr__() for dataset in self.datasets[:-1]]) + ", " + \
                   self.datasets[-1].__repr__() + \
                   ")"
        else:
            return "Concatenate({}xDatasets)".format(len(self.datasets))
