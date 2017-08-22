import torch.nn as nn
from ...utils.torch_utils import flatten_samples


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, channelwise=True):
        """
        Parameters
        ----------
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDiceLoss, self).__init__()
        self.channelwise = channelwise

    def forward(self, input, target):
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2. * (numerator / denominator)
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
            channelwise_loss = -2 * (numerator / denominator)
            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss
