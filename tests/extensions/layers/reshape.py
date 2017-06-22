import unittest


class TestReshape(unittest.TestCase):
    def test_as_matrix(self):
        import torch
        from inferno.extensions.layers.reshape import AsMatrix
        from torch.autograd import Variable

        input = Variable(torch.rand(10, 20, 1, 1))
        as_matrix = AsMatrix()
        output = as_matrix(input)
        self.assertEqual(list(output.size()), [10, 20])

    def test_flatten(self):
        import torch
        from inferno.extensions.layers.reshape import Flatten
        from torch.autograd import Variable

        input = Variable(torch.rand(10, 20, 2, 2))
        flatten = Flatten()
        output = flatten(input)
        self.assertEqual(list(output.size()), [10, 80])


if __name__ == '__main__':
    unittest.main()
