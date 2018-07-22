import unittest
import torch
import inferno.extensions.layers.building_blocks as bb
from inferno.extensions.layers.convolutional import ConvELU2D







class ResBlockTest(unittest.TestCase):

    def test_2D_simple(self):

        x = torch.autograd.Variable(torch.rand(1,3,64,15))
        model = bb.ResBlock(in_channels=3, out_channels=3, dim=2)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,3, 64, 15])

    def test_2D_simple2(self):


        x = torch.autograd.Variable(torch.rand(1,3,64,64))
        model = bb.ResBlock(in_channels=3, out_channels=6, dim=2)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,6, 64, 64])



if __name__ == '__main__':
    unittest.main()