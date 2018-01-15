import torch.nn as nn
import torch
from torch.autograd import Variable
# from ...utils.exceptions import assert_


class WeightedMSELoss(nn.Module):
    NEGATIVE_CLASS_WEIGHT = 1.

    def __init__(self, weight=None, positive_class_value=1., size_average=True):
        super(WeightedMSELoss, self).__init__()
        # assert_(positive_class_weight >= 0,
        #         "Positive class weight can't be less than zero, got {}."
        #         .format(positive_class_weight),
        #         ValueError)
        self.mse = nn.MSELoss(size_average=size_average)
        if weight is not None:
            assert torch.is_tensor(weight)
        self.weight = weight
        self.positive_class_value = positive_class_value

    def forward(self, input, target):
        if self.weight is None:
            return self.mse(input, target)
        else:
            # Get the positive class weight from the given positive and negative weights
            positive_class_weight = self.weight[1] / self.weight[0]
            # Get a mask
            positive_class_mask = target.data.eq(self.positive_class_value).type_as(target.data)
            # Get differential weights (positive_weight - negative_weight,
            # i.e. subtract 1, assuming the negative weight is gauged at 1)
            weight_differential = (positive_class_mask
                                   .mul_(positive_class_weight - self.NEGATIVE_CLASS_WEIGHT))
            # Get final weight by adding weight differential to a tensor with negative weights
            weights = weight_differential.add_(self.NEGATIVE_CLASS_WEIGHT)
            # `weights` should be positive if NEGATIVE_CLASS_WEIGHT is not messed with.
            sqrt_weights = Variable(weights.sqrt_(), requires_grad=False)
            return self.mse(input * sqrt_weights, target * sqrt_weights)


class WeightedBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedBCELoss, self).__init__()
        if weight is not None:
            assert torch.is_tensor(weight)
        self.weight = weight
        self.bce = nn.BCELoss()

    def forward(self, input, target):
        if self.weight is not None:
            self.bce.weight = torch.FloatTensor([self.weight[1] / self.weight[0]])
        return self.bce(input=input.view(-1), target=target.view(-1))
