import unittest
import torch.cuda as cuda
from inferno.utils.model_utils import ModelTester


class UNetTest(unittest.TestCase):
    def test_unet_2d(self):
        from inferno.extensions.models import UNet
        tester = ModelTester((1, 1, 256, 256), (1, 1, 256, 256))
        if cuda.is_available():
            tester.cuda()
        tester(UNet(1, 1, dim=2, initial_features=32))

    def test_unet_3d(self):
        from inferno.extensions.models import UNet
        tester = ModelTester((1, 1, 16, 64, 64), (1, 1, 16, 64, 64))
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(UNet(1, 1, dim=3, initial_features=8))


if __name__ == '__main__':
    unittest.main()
