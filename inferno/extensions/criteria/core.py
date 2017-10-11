import torch.nn as nn
from functools import reduce
from ...utils.exceptions import assert_, ShapeError, NotTorchModuleError


__all__ = ['Criteria', 'As2DCriterion']


class Criteria(nn.Module):
    """Aggregate multiple criteria to one."""
    def __init__(self, *criteria):
        super(Criteria, self).__init__()
        if len(criteria) == 1 and isinstance(criteria[0], (list, tuple)):
            criteria = list(criteria[0])
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


class As2DCriterion(nn.Module):
    """
    Makes a given criterion applicable on (N, C, H, W) prediction and (N, H, W) target tensors,
    if they're applicable to (N, C) prediction and (N,) target tensors .
    """
    def __init__(self, criterion):
        super(As2DCriterion, self).__init__()
        assert_(isinstance(criterion, nn.Module),
                "Criterion must be a module, got a {} instead."
                .format(type(criterion).__name__),
                NotTorchModuleError)
        self.criterion = criterion

    def forward(self, prediction, target):
        # Validate input
        assert_(prediction.dim() == 4, "`prediction` is expected to be a 4D tensor of shape "
                                       "(N, C, H, W), got a {}D "
                                       "tensor instead.".format(prediction.dim()),
                ShapeError)
        assert_(target.dim() == 3, "`target` is expected to be a 3D tensor of shape "
                                   "(N, H, W), got a {}D "
                                   "tensor instead.".format(target.dim()),
                ShapeError)
        # prediction is assumed to be NCHW, and target NHW.
        # this makes target (NHW,)
        target = target.contiguous().view(-1)
        # This makes prediction (N, H, W, C) --> (NHW, C)
        num_channels = prediction.size(1)
        prediction = prediction.permute(0, 2, 3, 1).contiguous().view(-1, num_channels)
        # Now, the criterion should be applicable as is
        loss = self.criterion(prediction, target)
        return loss
