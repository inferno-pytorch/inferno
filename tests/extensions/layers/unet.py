import unittest
import torch
import inferno.extensions.layers.unet as unet
from inferno.extensions.layers.convolutional import ConvELU2D







class ResidualUnetTest(unittest.TestCase):

    def test_2D_simple(self):

        in_channels = 3

        x = torch.autograd.Variable(torch.rand(1,in_channels,64,64))
        model = unet.ResBlockUnet(in_channels=in_channels, ndim=2)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,in_channels*2, 64, 64])

    def test_3D_simple(self):

        in_channels = 3

        x = torch.autograd.Variable(torch.rand(1,in_channels,64,32,64))
        model = unet.ResBlockUnet(in_channels=in_channels, ndim=3)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,in_channels*2, 64,32, 64])


if __name__ == '__main__':
    unittest.main()