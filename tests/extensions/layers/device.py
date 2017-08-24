import unittest
from inferno.extensions.layers.device import DeviceTransfer
import torch
from torch.autograd import Variable


class TransferTest(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()