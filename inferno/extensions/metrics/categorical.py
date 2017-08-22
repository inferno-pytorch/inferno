import torch
from .base import Metric
from ...utils.torch_utils import flatten_samples
from ...utils.exceptions import assert_


class CategoricalError(Metric):
    """Categorical error."""
    def __init__(self, aggregation_mode='mean'):
        assert aggregation_mode in ['mean', 'sum']
        self.aggregation_mode = aggregation_mode

    def forward(self, prediction, target):
        # Check if prediction is binary or not
        is_binary = len(prediction.size()) == 1 or prediction.size(1) == 1

        if len(target.size()) > 1:
            target = target.squeeze(1)
        assert len(target.size()) == 1

        if is_binary:
            # Binary classification
            prediction = prediction > 0.5
            incorrect = prediction.type_as(target).ne(target).float()
            if self.aggregation_mode == 'mean':
                return incorrect.mean()
            else:
                return incorrect.sum()
        else:
            # Multiclass classificiation
            _, predicted_class = torch.max(prediction, 1)
            incorrect = predicted_class.squeeze(1).type_as(target).ne(target).float()
            if self.aggregation_mode == 'mean':
                return incorrect.mean()
            else:
                return incorrect.sum()


class IOU(Metric):
    """Intersection over Union. """
    def __init__(self, eps=1e-6):
        super(IOU, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        # Assume that is one of:
        #   prediction.shape = (N, C, H, W)
        #   prediction.shape = (N, C, D, H, W)
        #   prediction.shape = (N, C)
        # The corresponding target shapes are:
        #   target.shape = (N, H, W)
        #   target.shape = (N, D, H, W)
        #   target.shape = (N,)
        # First, reshape prediction to (C, -1)
        flattened_prediction = flatten_samples(prediction)
        # Reshape target to (1, -1) for it to work with scatter
        flattened_target = target.view(1, -1)
        # Convert target to onehot with shape (C, -1)
        num_classes, num_samples = flattened_prediction.size()
        # Make sure the target is consistent
        assert_(target.max() < num_classes)
        onehot_targets = flattened_prediction\
            .new(num_classes, num_samples)\
            .zero_()\
            .scatter_(0, flattened_target, 1)
        # Now to compute the IOU = (a * b).sum()/(a**2 + b**2 - a * b).sum()
        # = (a * b).sum()/((a - b)**2).sum()
        # We sum over all samples and average over all classes
        numerator = (flattened_prediction * onehot_targets).sum(-1).mean()
        denominator = \
            (flattened_prediction - onehot_targets).pow_(2).clamp_(min=self.eps).sum(-1).mean()
        iou = (numerator / denominator)
        return iou


class NegativeIOU(IOU):
    def forward(self, prediction, target):
        return -1 * super(NegativeIOU, self).forward(prediction, target)
