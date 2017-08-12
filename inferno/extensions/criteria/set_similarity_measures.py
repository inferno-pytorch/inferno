import torch.nn as nn


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target.
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

    @staticmethod
    def flatten_samples(tensor_or_variable):
        """
        Flattens a tensor such that the channel axis is first and the sample axis is second.
        A (N, C, H, W) tensor would be flattened to a (C, N * H * W) tensor, for instance.
        """
        # Get number of channels
        num_channels = tensor_or_variable.size(1)
        # Permute the channel axis to first
        permute_axes = list(range(tensor_or_variable.dim()))
        permute_axes[0], permute_axes[1] = permute_axes[1], permute_axes[0]
        # For input shape (say) NCHW, this should have the shape CNHW
        permuted = tensor_or_variable.permute(*permute_axes).contiguous()
        # Now flatten out all but the first axis and return
        flattened = permuted.view(num_channels, -1)
        return flattened

    def forward(self, input, target):
        if not self.channelwise:
            numerator = (input * target).sum()
            denominator = (input * input).sum() + (target * target).sum()
            loss = -2. * (numerator / denominator)
        else:
            # TODO This should be compatible with Pytorch 0.2, but check
            # Flatten input and target to have the shape (C, N),
            # where N is the number of samples
            input = self.flatten_samples(input)
            target = self.flatten_samples(target)
            # Compute numerator and denominator (by summing over samples and
            # leaving the channels intact)
            numerator = (input * target).sum(-1)
            denominator = (input * input).sum(-1) + (target * target).sum(-1)
            channelwise_loss = -2 * (numerator / denominator)
            # Sum over the channels to compute the total loss
            loss = channelwise_loss.sum()
        return loss
