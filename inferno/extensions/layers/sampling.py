import torch.nn as nn

__all__ = ['AnisotropicUpsample', 'AnisotropicPool']


class AnisotropicUpsample(nn.Module):
    def __init__(self, scale_factor):
        super(AnisotropicUpsample, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=scale_factor)

    def forward(self, input):
        # input is 3D of shape NCDHW
        N, C, D, H, W = input.size()
        # Fold C and D axes in one
        folded = input.view(N, C * D, H, W)
        # Upsample
        upsampled = self.upsampler(folded)
        # Unfold out the C and D axes
        unfolded = upsampled.view(N, C, D,
                                  self.upsampler.scale_factor * H,
                                  self.upsampler.scale_factor * W)
        # Done
        return unfolded


class AnisotropicPool(nn.MaxPool3d):
    def __init__(self, downscale_factor):
        ds = downscale_factor
        super(AnisotropicPool, self).__init__(kernel_size=(1, ds + 1, ds + 1),
                                              stride=(1, ds, ds),
                                              padding=(0, 1, 1))
