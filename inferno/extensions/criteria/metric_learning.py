import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['TripletLoss']

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0, p=2, swap=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.swap = swap
    def forward(self, triplet, target=None):
        #inferno trainer assumes every loss has a target; target not used here
        # we might or might not have the batch dim here
        assert len(triplet) == 3 or len(triplet[0]) == 3
        if len(triplet) == 3:
            triplet = triplet.unsqueeze(0)
        losses = [F.triplet_margin_loss(batch[0], batch[1], batch[2],
                                        margin=self.margin, swap=self.swap)
                  for batch in triplet]
        return sum(losses) / len(triplet)
