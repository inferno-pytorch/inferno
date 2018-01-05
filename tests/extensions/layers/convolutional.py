import unittest
import torch
from inferno.utils.model_utils import ModelTester


class TestConvolutional(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "GPU not available.")
    def test_bn_relu_depthwise_conv2d_pyinn(self):
        from inferno.extensions.layers.convolutional import BNReLUDepthwiseConv2D
        model = BNReLUDepthwiseConv2D(10, 'auto', 3)
        ModelTester((1, 10, 100, 100),
                    (1, 10, 100, 100)).cuda()(model)
        self.assertTrue(model.depthwise)
        self.assertEqual(model.conv.groups, 10)


if __name__ == '__main__':
    unittest.main()
