import torch.nn as nn


class UpsampleNearest(nn.Upsample):
    def __init__(self, scale_factor=2):
        super(UpsampleNearest, self).__init__(scale_factor=scale_factor, mode='nearest')


class UpsampleBilinear(nn.Upsample):
    def __init__(self, scale_factor=2):
        super(UpsampleBilinear, self).__init__(scale_factor=scale_factor, mode='bilinear')


class UpsampleTrilinear(nn.Upsample):
    def __init__(self, scale_factor=2):
        super(UpsampleTrilinear, self).__init__(scale_factor=scale_factor, mode='trilinear')
