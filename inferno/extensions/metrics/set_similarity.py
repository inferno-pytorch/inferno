import torch.nn as nn
from ...utils.torch_utils import flatten_samples

__all__ = ['SorensenDice']


class SorensenDice(nn.Module):
    """
    Computes a Sorensen-Dice similarity
    between the input and the target. For both inputs and targets it must be the case that
    `input_or_target.size(1) = num_channels`.
    """
    def __init__(self, channelwise=True, eps=1e-6):
        """
        Parameters
        ----------
        channelwise : bool
            Whether to apply the loss channelwise and sum the results (True)
            or to apply it on all channels jointly (False).
        """
        super(SorensenDice, self).__init__()
        self.channelwise = channelwise
        self.eps = eps

    def forward(self, input, target):
        """
        input:      torch.FloatTensor or torch.cuda.FloatTensor
        target:     torch.FloatTensor or torch.cuda.FloatTensor

        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        assert input.size() == target.size(), "%s, %s" % (str(input.shape), str(target.shape))
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            score = 1. - 2. * (numerator / denominator.clamp(min=self.eps))
        else:
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            n_channels = input.size(1)
            input = flatten_samples(input)
            target = flatten_samples(target)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_score = 2 * (numerator / denominator.clamp(min=self.eps))
            # Sum over the channels to compute the total loss
            score = n_channels - channelwise_score.sum()
        # we want 0 to be the optimal metric -> invert the score
        return score
