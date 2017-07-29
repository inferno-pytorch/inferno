import torch.nn as nn
from functools import reduce


class Criteria(nn.Module):
    """Aggregate multiple criteria to one."""
    def __init__(self, *criteria):
        super(Criteria, self).__init__()
        if len(criteria) == 1 and isinstance(criteria[0], (list, tuple)):
            criteria = list(criteria)
        else:
            criteria = list(criteria)
        # Validate criteria
        assert all([isinstance(criterion, nn.Module) for criterion in criteria]), \
            "Criterion must be a torch module."
        self.criteria = criteria

    def forward(self, prediction, target):
        assert isinstance(prediction, (list, tuple)), \
            "`prediction` must be a list or a tuple, got {} instead."\
                .format(type(prediction).__name__)
        assert isinstance(target, (list, tuple)), \
            "`prediction` must be a list or a tuple, got {} instead." \
                .format(type(target).__name__)
        assert len(prediction) == len(target), \
            "Number of predictions must equal the number of targets. " \
            "Got {} predictions but {} targets.".format(len(prediction), len(target))
        # Compute losses
        losses = [criterion(prediction, target)
                  for _prediction, _target, criterion in zip(prediction, target, self.criteria)]
        # Aggegate losses
        loss = reduce(lambda x, y: x + y, losses)
        # Done
        return loss
