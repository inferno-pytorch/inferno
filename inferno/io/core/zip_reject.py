from .zip import Zip


class ZipReject(Zip):
    """
    ZipReject: Extends zip by the functionality of rejecting samples that don't fullfill
    a specified rejection criterion.
    ----------
    Parameter:
    datasets, sync, transforms: Parameter of Zip.
    rejection_dataset: id of dataset that is used to determine whether batch should be rejected.
    rejection_criterion: criterion for rejection of batch. Must be a callable that accepts a single
                         array / tensor and returns True if the corresponding batch should be rejected.
    """
    def __init__(self, *datasets, sync=False, transforms=None,
                 rejection_dataset, rejection_criterion):
        super(ZipReject, self).__init__(*datasets, sync=sync, transforms=transforms)

        assert rejection_dataset < len(datasets)
        self.rejection_dataset = rejection_dataset
        assert callable(rejection_criterion)
        self.rejection_criterion = rejection_criterion  # return true if fetched should be rejected

    # we increase the index until a valid batch of 'rejection_dataset' is found
    def __getitem__(self, index):

        assert index < len(self)
        index_ = index

        # if we have a rejection dataset, check if the rejection criterion is fulfilled
        # and update the index
        if self.rejection_dataset is not None:

            # we only fetch the dataset which has the rejection criterion
            # and only fetch all datasets when a valid index is found
            rejection_fetched = self.datasets[self.rejection_dataset][index_]
            loop_counter = 0
            while self.rejection_criterion(rejection_fetched):
                index_ = (index_ + 1) % len(self)
                rejection_fetched = self.datasets[self.rejection_dataset][index_]
                loop_counter += 1
                if loop_counter >= len(self):
                    raise RuntimeError("ZipReject: No valid batch was found!")

            # fetch all other datasets and concatnate them with the valid rejection_fetch
            fetched = [dataset[index_] for dataset in self.datasets[:self.rejection_dataset]] + \
                      [rejection_fetched] + \
                      [dataset[index_] for dataset in self.datasets[self.rejection_dataset + 1:]]
        else:
            fetched = [dataset[index_] for dataset in self.datasets]

        # apply transforms if present
        if self.transforms is not None:
            assert callable(self.transforms)
            fetched = self.transforms(*fetched)

        return fetched

    def __repr__(self):
        if len(self.datasets) > 3:
            return "ZipReject({}xDatasets)".format(len(self.datasets))
        else:
            return "ZipReject(" + \
                   ", ".join([dataset.__repr__() for dataset in self.datasets[:-1]]) + ", " + \
                   self.datasets[-1].__repr__() + \
                   ')'

