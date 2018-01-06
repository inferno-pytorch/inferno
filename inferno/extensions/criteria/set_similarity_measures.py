import torch.nn as nn
from ...utils.torch_utils import flatten_samples
from torch.autograd import Variable

__all__ = ['SorensenDiceLoss', 'GeneralizedDiceLoss', 'TverskyLoss']


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
            # FIXME weight does not do what I expect:
            # instead of weighting the individual classes, it weights the channels.
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

    This version works for multiple classes and expects inputs for every class (e.g. softmax output) and
    one-hot targets for every class.
    """
    def __init__(self, eps=1e-6):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, input, target):
        # Flatten input and target to have the shape (C, N),
        # where N is the number of samples
        input = flatten_samples(input)
        target = flatten_samples(target)

        # Find classes weights:
        sum_targets = target.sum(-1)
        class_weigths = 1. / (sum_targets * sum_targets).clamp(min=self.eps)

        # # Compute generalized Dice loss:
        numer = ((input * target).sum(-1) * class_weigths).sum()
        denom = ((input + target).sum(-1) * class_weigths).sum()

        loss = 1. - 2. * numer / denom.clamp(min=self.eps)
        return loss


class TverskyLoss(nn.Module):
    """
    Computes a loss scalar according to Salehi et al., which generalizes the Dice loss.
    It has to parameters, alpha and beta, which weight the False Positives and False Negatives, respectively.
    For alpha = beta = 0.5 TverslyLoss reduces to Dice Loss.
    In Salehis paper beta = 0.7, alpha = 1 - beta = 0.3 are optimal for very unbalanced data.
    """
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        """
        Parameters
        ----------
        alpha: weight for the FPs
        beta:  weight for the FNs
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, input, target):
        '''input and target are respectively a tensor of the shape (N,*) with the batch_size N
        the output is the mean over the loss of each batch dimension'''

        batch_size = input.size(0)
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)

        numerator = (input * target).sum(dim=1)
        denominator = (input * target).sum(dim=1) + self.alpha * ((1. - target) * input).sum(dim=1) + \
            self.beta * ((1. - input) * target).sum(dim=1)

        losses = -numerator / denominator.clamp(min=self.eps)
        loss = losses.sum() / batch_size
        return loss
