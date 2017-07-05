import torch.nn as nn


class SorensenDiceLoss(nn.Module):
    """
    Computes a loss scalar, which when minimized maximizes the Sorensen-Dice similarity
    between the input and the target.
    """
    def forward(self, input, target):
        numerator = (input * target).sum()
        denominator = (input * input).sum() + (target * target).sum()
        loss = -2. * (numerator / denominator)
        return loss
