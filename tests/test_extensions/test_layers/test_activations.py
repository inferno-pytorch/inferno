import unittest
import torch
import inferno.extensions.layers.activations as activations


class ActivationTest(unittest.TestCase):
    def test_selu(self):
        x = torch.rand(100)
        y = activations.SELU()(x)
        self.assertEqual(list(x.size()), list(y.size()))


if __name__ == '__main__':
    unittest.main()
