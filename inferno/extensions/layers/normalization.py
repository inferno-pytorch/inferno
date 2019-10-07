import torch.nn as nn


class BatchNormND(nn.Module):
    def __init__(self, dim, num_features, 
                 eps=1e-5, momentum=0.1, 
                 affine=True,track_running_stats=True):
        super(BatchNormND, self).__init__()
        assert dim in [1, 2, 3]
        self.bn = getattr(nn, 'BatchNorm{}d'.format(dim))(num_features=num_features,
            eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    def forward(self, x):
        return self.bn(x)