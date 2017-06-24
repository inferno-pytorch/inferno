from torch.utils.data.dataset import Dataset


class SyncableDataset(Dataset):
    def __init__(self):
        self.base_sequence = None

    def sync_with(self, dataset):
        if hasattr(dataset, 'base_sequence'):
            self.base_sequence = dataset.base_sequence
        return self

    def __len__(self):
        if self.base_sequence is None:
            raise RuntimeError("Class {} does not specify a base sequence. Either specify "
                               "one by assigning to self.base_sequence or override the "
                               "__len__ method.".format(self.__class__.__name__))
        else:
            return len(self.base_sequence)
