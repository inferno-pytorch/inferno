import unittest
import torch
import inferno.extensions.layers.building_blocks as bb
from inferno.extensions.layers.convolutional import ConvELU2D







class ResBlockTest(unittest.TestCase):

    def test_2D_simple_(self):

        x = torch.autograd.Variable(torch.rand(1,3,64,15))
        model = bb.ResBlock(in_channels=3, out_channels=3, dim=2)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,3, 64, 15])

    def test_3D_simple_(self):

        x = torch.autograd.Variable(torch.rand(1,3,20, 64,15))
        model = bb.ResBlock(in_channels=3, out_channels=3, dim=3)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,3, 20, 64, 15])

    def test_2D_simple_2(self):


        x = torch.autograd.Variable(torch.rand(1,3,64,64))
        model = bb.ResBlock(in_channels=3, out_channels=6, dim=2)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,6, 64, 64])

    def test_2D_simple_3(self):


        x = torch.autograd.Variable(torch.rand(1,3,64,64))
        model = bb.ResBlock(in_channels=3, out_channels=6, dim=2, size=4)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,6, 64, 64])

    def test_2D_simple_4(self):


        x = torch.autograd.Variable(torch.rand(1,6,64,64))
        model = bb.ResBlock(in_channels=6, out_channels=6, dim=2, size=4,
            force_skip_op=True)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,6, 64, 64])

    def test_2D_simple_5(self):


        x = torch.autograd.Variable(torch.rand(1,6,64,64))
        model = bb.ResBlock(in_channels=6, batchnorm=False, out_channels=6, dim=2, size=4,
            force_skip_op=True)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,6, 64, 64])

    def test_2D_simple_6(self):


        x = torch.autograd.Variable(torch.rand(1,6,64,64))
        model = bb.ResBlock(in_channels=6, batchnorm=False, out_channels=6, dim=2, size=4,
            force_skip_op=True, activated=False)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,6, 64, 64])


    def test_3D_simple_6(self):


        x = torch.autograd.Variable(torch.rand(1,6,64,64, 20))
        model = bb.ResBlock(in_channels=6, batchnorm=False, out_channels=6, dim=3, size=4,
            force_skip_op=True, activated=False)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,6, 64, 64, 20])

if __name__ == '__main__':
    unittest.main()
