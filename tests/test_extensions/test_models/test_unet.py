import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import ModelTester, MultiscaleModelTester
from inferno.extensions.models import UNet

class _MultiscaleUNet(UNet):
    def conv_op_factory(self, in_channels, out_channels, part, index):
        return super(_MultiscaleUNet, self).conv_op_factory(in_channels, out_channels, part, index)[0], True

    def forward(self, input):
        x = self._initial_conv(input)
        x = list(super(UNet, self).forward(x))
        x[-1] = self._output(x[-1])
        return tuple(x)


class UNetTest(unittest.TestCase):
    def test_unet_2d(self):
        tester = ModelTester((1, 1, 256, 256), (1, 1, 256, 256))
        if cuda.is_available():
            tester.cuda()
        tester(UNet(1, 1, dim=2, initial_features=32))

    def test_unet_3d(self):
        tester = ModelTester((1, 1, 16, 64, 64), (1, 1, 16, 64, 64))
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(UNet(1, 1, dim=3, initial_features=8))

    def test_monochannel_unet_3d(self):
        nc = 2
        class _UNetMonochannel(_MultiscaleUNet):
            def _get_num_channels(self, depth):
                return nc

        shapes = [(1, nc, 16, 64, 64), (1, nc, 8, 32, 32), (1, nc, 4, 16, 16), (1, nc, 2, 8, 8), (1, nc, 1, 4, 4),
                  (1, nc, 2, 8, 8), (1, nc, 4, 16, 16), (1, nc, 8, 32, 32), (1, 1, 16, 64, 64)]
        tester = MultiscaleModelTester((1, 1, 16, 64, 64), shapes)
        if cuda.is_available():
            tester.cuda()
        tester(_UNetMonochannel(1, 1, dim=3, initial_features=8))

    def test_inverse_pyramid_unet_2d(self):
        class _UNetInversePyramid(_MultiscaleUNet):
            def _get_num_channels(self, depth):
                return [13, 12, 11][depth - 1]

        shapes = [(1, 13, 16, 64), (1, 12, 8, 32), (1, 11, 4, 16), (1, 11, 2, 8),
                  (1, 12, 4, 16), (1, 13, 8, 32), (1, 1, 16, 64)]
        tester = MultiscaleModelTester((1, 1, 16, 64), shapes)
        if cuda.is_available():
            tester.cuda()
        tester(_UNetInversePyramid(1, 1, dim=2, depth=3, initial_features=8))


if __name__ == '__main__':
    unittest.main()
