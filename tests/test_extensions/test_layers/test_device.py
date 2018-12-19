import unittest
from inferno.extensions.layers.device import DeviceTransfer, OnDevice
import torch
from torch.autograd import Variable


class TransferTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "GPU not available.")
    def test_device_transfer(self):
        if not torch.cuda.is_available():
            return
        # Build transfer model
        transfer = DeviceTransfer('cpu')
        x = Variable(torch.rand(10, 10).cuda(), requires_grad=True)
        y = transfer(x)
        loss = y.mean()
        loss.backward()
        self.assertFalse(y.data.is_cuda)
        self.assertIsNotNone(x.grad)
        self.assertTrue(x.grad.data.is_cuda)

    @unittest.skipIf(not torch.cuda.is_available(), "GPU not available.")
    def test_on_device(self):
        if not torch.cuda.is_available():
            return
        # Build variable on the GPU
        x = Variable(torch.rand(1, 10))
        # Build model over multiple devices
        multi_device_model = torch.nn.Sequential(OnDevice(torch.nn.Linear(10, 10), 'cuda'),
                                                 OnDevice(torch.nn.Linear(10, 10), 'cpu'))
        y = multi_device_model(x)
        self.assertIsInstance(y.data, torch.FloatTensor)

if __name__ == '__main__':
    unittest.main()