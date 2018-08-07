import unittest
import torch
import inferno.extensions.layers.unet as unet
from inferno.extensions.layers.convolutional import ConvELU2D




class MyTestUNet(unet.UNetBase):

    def __init__(self, dim, in_channels, out_channels, depth):


        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.unet_kwargs = dict(depth=depth)

        # unit test related
        self.factory_calls_counter = {
            'up' : 0, 'bottom' : 0, 'down' : 0
        } 

        super(MyTestUNet, self).__init__(
            in_channels=in_channels, 
            dim=dim,
            out_channels=out_channels, 
            **self.unet_kwargs
        )



    def conv_op_factory(self, in_channels, out_channels, part, index):
        self.factory_calls_counter[part] += 1
        if self.dim  == 2:
            return ConvELU2D(kernel_size=3, in_channels=in_channels,
                out_channels=out_channels)
        else:
            return ConvELU3D(kernel_size=3, in_channels=in_channels,
                out_channels=out_channels)

class CustomUnetTest(unittest.TestCase):

    def test_2D_simple(self):
        in_channels = 3

        for depth in [1,2,3,4,5]:

            x = torch.autograd.Variable(torch.rand(1,in_channels,2**depth,2**depth))
            model = MyTestUNet(in_channels=in_channels, out_channels=6, dim=2, depth=depth)


            self.assertEqual(model.factory_calls_counter['down'],   depth)
            self.assertEqual(model.factory_calls_counter['bottom'],     1)
            self.assertEqual(model.factory_calls_counter['down'],   depth)

class ResidualUnetTest(unittest.TestCase):

    def test_2D_simple(self):

        in_channels = 3

        x = torch.autograd.Variable(torch.rand(1,in_channels,64,56))
        model = unet.ResBlockUNet(in_channels=in_channels, out_channels=6, dim=2)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,in_channels*2, 64, 56])

    def test_3D_simple(self):

        in_channels = 3

        x = torch.autograd.Variable(torch.rand(1,in_channels,64,32,80))
        model = unet.ResBlockUNet(in_channels=in_channels, out_channels=6, dim=3)
        xx = model(x)
        out_size = xx.size()
        self.assertEqual(list(out_size), [1,in_channels*2, 64,32, 80])



    def test_2D_side_out(self):

        depth = 3
        in_channels = 3

        x = torch.autograd.Variable(torch.rand(1,in_channels,64,32))
        model = unet.ResBlockUNet(in_channels=in_channels, out_channels=8, dim=2, 
            side_out_parts=['bottom','up'], unet_kwargs=dict(depth=depth))

        out_list = model(x)
        self.assertEqual(len(out_list), depth + 1)


        self.assertEqual(list(out_list[0].size()), [1,24, 8, 4])
        self.assertEqual(list(out_list[1].size()), [1,12, 16, 8])
        self.assertEqual(list(out_list[2].size()), [1, 6, 32, 16])
        self.assertEqual(list(out_list[3].size()), [1, 8, 64, 32])


    def test_2D_side_out2(self):

        depth = 3
        in_channels = 3

        x = torch.autograd.Variable(torch.rand(1,in_channels,64,32))
        model = unet.ResBlockUNet(in_channels=in_channels, out_channels=8, dim=2, 
            side_out_parts=['up'], unet_kwargs=dict(depth=depth))

        out_list = model(x)
        self.assertEqual(len(out_list), depth )


        self.assertEqual(list(out_list[0].size()), [1,12, 16, 8])
        self.assertEqual(list(out_list[1].size()), [1, 6, 32, 16])
        self.assertEqual(list(out_list[2].size()), [1, 8, 64, 32])


    def test_2D_side_out2(self):

        depth = 3
        in_channels = 3

        x = torch.autograd.Variable(torch.rand(1,in_channels,64,32))
        model = unet.ResBlockUNet(in_channels=in_channels, out_channels=8, dim=2, 
            side_out_parts=['down'], unet_kwargs=dict(depth=depth))

        out_list = model(x)
        self.assertEqual(len(out_list), depth  + 1 )

        self.assertEqual(list(out_list[0].size()), [1, 6, 64, 32])
        self.assertEqual(list(out_list[1].size()), [1, 12, 32, 16])
        self.assertEqual(list(out_list[2].size()), [1, 24, 16, 8])
            
        # the actual output
        self.assertEqual(list(out_list[3].size()), [1, 8, 64, 32])
        

if __name__ == '__main__':
    unittest.main()