import unittest
import torch
import torch.cuda as cuda
from inferno.utils.model_utils import ModelTester


class ResUNetTest(unittest.TestCase):
    def test_res_unet_2d(self):
        from inferno.extensions.models import ResBlockUNet
        tester = ModelTester((1, 1, 256, 256), (1, 1, 256, 256))
        if cuda.is_available():
            tester.cuda()
        tester(ResBlockUNet(in_channels=1, out_channels=1, dim=2))

    def test_res_unet_3d(self):
        from inferno.extensions.models import ResBlockUNet
        tester = ModelTester((1, 1, 16, 64, 64), (1, 1, 16, 64, 64))
        if cuda.is_available():
            tester.cuda()
        # test default unet 3d
        tester(ResBlockUNet(in_channels=1, out_channels=1, dim=3))

    def test_2d_side_out_bot_up(self):
        from inferno.extensions.models import ResBlockUNet
        depth = 3
        in_channels = 3

        x = torch.rand(1, in_channels, 64, 32)
        model = ResBlockUNet(in_channels=in_channels,
                             out_channels=8, dim=2,
                             side_out_parts=['bottom','up'],
                             unet_kwargs=dict(depth=depth))

        out_list = model(x)
        self.assertEqual(len(out_list), depth + 1)

        self.assertEqual(list(out_list[0].size()), [1, 24, 8, 4])
        self.assertEqual(list(out_list[1].size()), [1, 12, 16, 8])
        self.assertEqual(list(out_list[2].size()), [1, 6, 32, 16])
        self.assertEqual(list(out_list[3].size()), [1, 8, 64, 32])

    def test_2d_side_out_up(self):
        from inferno.extensions.models import ResBlockUNet
        depth = 3
        in_channels = 3

        x = torch.rand(1, in_channels, 64, 32)
        model = ResBlockUNet(in_channels=in_channels,
                             out_channels=8, dim=2,
                             side_out_parts=['up'],
                             unet_kwargs=dict(depth=depth))

        out_list = model(x)
        self.assertEqual(len(out_list), depth)

        self.assertEqual(list(out_list[0].size()), [1,12, 16, 8])
        self.assertEqual(list(out_list[1].size()), [1, 6, 32, 16])
        self.assertEqual(list(out_list[2].size()), [1, 8, 64, 32])

    def test_2d_side_out_down(self):
        from inferno.extensions.models import ResBlockUNet
        depth = 3
        in_channels = 3

        x = torch.rand(1, in_channels, 64, 32)
        model = ResBlockUNet(in_channels=in_channels,
                             out_channels=8, dim=2,
                             side_out_parts=['down'],
                             unet_kwargs=dict(depth=depth))

        out_list = model(x)
        self.assertEqual(len(out_list), depth  + 1)

        self.assertEqual(list(out_list[0].size()), [1, 6, 64, 32])
        self.assertEqual(list(out_list[1].size()), [1, 12, 32, 16])
        self.assertEqual(list(out_list[2].size()), [1, 24, 16, 8])

        # the actual output
        self.assertEqual(list(out_list[3].size()), [1, 8, 64, 32])


if __name__ == '__main__':
    unittest.main()
