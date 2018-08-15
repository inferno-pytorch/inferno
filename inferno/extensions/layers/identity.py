import torch.nn as nn
__all__ = ['identity']
_all = __all__

class Identity(nn.Module):  
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x