import torch.nn as nn
from ...utils.torch_utils import flatten_samples
from torch.autograd import Variable

__all__ = ['SorensenDiceLoss', 'GeneralizedDiceLoss']


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, weight=None, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        weight : torch.FloatTensor or torch.cuda.FloatTensor
            Class weights. Applies only if `channelwise = True`.
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2. * (numerator / denominator.clamp(min=self.eps))
        else:
            # TODO This should be compatible with Pytorch 0.2, but check
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            input = flatten_samples(input)
            target = flatten_samples(target)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator.clamp(min=self.eps))
            if self.weight is not None:
                # With pytorch < 0.2, channelwise_loss.size = (C, 1).
                if channelwise_loss.dim() == 2:
                    channelwise_loss = channelwise_loss.squeeze(1)
                # Wrap weights in a variable
                weight = Variable(self.weight, requires_grad=False)
                # Apply weight
                channelwise_loss = weight * channelwise_loss
            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss

class GeneralizedDiceLoss(nn.Module):
    """
    Computes the scalar Generalized Dice Loss defined in https://arxiv.org/abs/1707.03237

    This version works for multiple classes and expects predictions for every class (e.g. softmax output) and
    one-hot targets for every class.
    """
    def __init__(self, eps=1e-6):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        # Flatten input and target to have the shape (C, N),
        # where N is the number of samples
        prediction = flatten_samples(prediction)
        target = flatten_samples(target)

        # Find classes weights:
        sum_targets = target.sum(-1)
        class_weigths = 1. / (sum_targets*sum_targets).clamp(min=self.eps)

        # # Compute generalized Dice loss:
        numer = ((prediction*target).sum(-1) * class_weigths).sum()
        denom = ((prediction+target).sum(-1) * class_weigths).sum()

        loss = 1. - 2. * numer / denom.clamp(min=self.eps)

        return loss
