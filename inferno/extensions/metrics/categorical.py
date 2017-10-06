import torch
from .base import Metric
from ...utils.torch_utils import flatten_samples, is_label_tensor
from ...utils.exceptions import assert_, DTypeError, ShapeError


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
            if predicted_class.dim() == prediction.dim():
                # Support for Pytorch 0.1.12
                predicted_class = predicted_class.squeeze(1)
            incorrect = predicted_class.type_as(target).ne(target).float()
            if self.aggregation_mode == 'mean':
                return incorrect.mean()
            else:
                return incorrect.sum()


class IOU(Metric):
    """Intersection over Union. """
    def __init__(self, ignore_class=None, sharpen_prediction=False, eps=1e-6):
        super(IOU, self).__init__()
        self.eps = eps
        self.ignore_class = ignore_class
        self.sharpen_prediction = sharpen_prediction

    def forward(self, prediction, target):
        # Assume that is one of:
        #   prediction.shape = (N, C, H, W)
        #   prediction.shape = (N, C, D, H, W)
        #   prediction.shape = (N, C)
        # The corresponding target shapes are either:
        #   target.shape = (N, H, W)
        #   target.shape = (N, D, H, W)
        #   target.shape = (N,)
        # Or:
        #   target.shape = (N, C, H, W)
        #   target.shape = (N, C, D, H, W)
        #   target.shape = (N, C)
        # First, reshape prediction to (C, -1)
        flattened_prediction = flatten_samples(prediction)
        # Take measurements
        num_classes, num_samples = flattened_prediction.size()
        # We need to figure out if the target is a int label tensor or a onehot tensor.
        # The former always has one dimension less, so
        if target.dim() == (prediction.dim() - 1):
            # Labels, we need to go one hot
            # Make sure it's a label
            assert_(is_label_tensor(target),
                    "Target must be a label tensor (of dtype long) if it has one "
                    "dimension less than the prediction.",
                    DTypeError)
            # Reshape target to (1, -1) for it to work with scatter
            flattened_target = target.view(1, -1)
            # Convert target to onehot with shape (C, -1)
            # Make sure the target is consistent
            assert_(target.max() < num_classes)
            onehot_targets = flattened_prediction \
                .new(num_classes, num_samples) \
                .zero_() \
                .scatter_(0, flattened_target, 1)
        elif target.dim() == prediction.dim():
            # Onehot, nothing to do except flatten
            onehot_targets = flatten_samples(target)
        else:
            raise ShapeError("Target must have the same number of dimensions as the "
                             "prediction, or one less. Got target.dim() = {} but "
                             "prediction.dim() = {}.".format(target.dim(), prediction.dim()))
        # Sharpen prediction if required to. Sharpening in this sense means to replace
        # the max predicted probability with 1.
        if self.sharpen_prediction:
            _, predicted_classes = torch.max(flattened_prediction, 0)
            # Case for pytorch 0.2, where predicted_classes is (N,) instead of (1, N)
            if predicted_classes.dim() == 1:
                predicted_classes = predicted_classes.view(1, -1)
            # Scatter
            flattened_prediction = flattened_prediction\
                .new(num_classes, num_samples).zero_().scatter_(0, predicted_classes, 1)
        # Now to compute the IOU = (a * b).sum()/(a**2 + b**2 - a * b).sum()
        # We sum over all samples to obtain a classwise iou
        numerator = (flattened_prediction * onehot_targets).sum(-1)
        denominator = \
            flattened_prediction.sub_(onehot_targets).pow_(2).clamp_(min=self.eps).sum(-1) + \
            numerator
        classwise_iou = numerator.div_(denominator)
        # If we're ignoring a class, don't count its contribution to the mean
        if self.ignore_class is not None:
            ignore_class = self.ignore_class \
                if self.ignore_class != -1 else onehot_targets.size(0) - 1
            assert_(ignore_class < onehot_targets.size(0),
                    "`ignore_class` = {} must be at least one less than the number "
                    "of classes = {}.".format(ignore_class, onehot_targets.size(0)),
                    ValueError)
            num_classes = onehot_targets.size(0)
            dont_ignore_class = list(range(num_classes))
            dont_ignore_class.pop(ignore_class)
            if classwise_iou.is_cuda:
                dont_ignore_class = \
                    torch.LongTensor(dont_ignore_class).cuda(classwise_iou.get_device())
            else:
                dont_ignore_class = torch.LongTensor(dont_ignore_class)
            iou = classwise_iou[dont_ignore_class].mean()
        else:
            iou = classwise_iou.mean()
        return iou


class NegativeIOU(IOU):
    def forward(self, prediction, target):
        return -1 * super(NegativeIOU, self).forward(prediction, target)
