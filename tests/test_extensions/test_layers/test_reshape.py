import unittest
import torch


class TestReshape(unittest.TestCase):
    def _get_input_variable(self, *shape):
        return torch.rand(*shape)

    def test_as_matrix(self):
        from inferno.extensions.layers.reshape import AsMatrix

        input = self._get_input_variable(10, 20, 1, 1)
        as_matrix = AsMatrix()
        output = as_matrix(input)
        self.assertEqual(list(output.size()), [10, 20])

    def test_flatten(self):
        from inferno.extensions.layers.reshape import Flatten

        input = self._get_input_variable(10, 20, 2, 2)
        flatten = Flatten()
        output = flatten(input)
        self.assertEqual(list(output.size()), [10, 80])

    def test_as_2d(self):
        from inferno.extensions.layers.reshape import As2D

        as_2d = As2D()

        output_shape = as_2d(self._get_input_variable(10, 20, 3, 30, 30)).size()
        self.assertEqual(list(output_shape), [10, 60, 30, 30])

        output_shape = as_2d(self._get_input_variable(10, 20, 30, 30)).size()
        self.assertEqual(list(output_shape), [10, 20, 30, 30])

        output_shape = as_2d(self._get_input_variable(10, 20)).size()
        self.assertEqual(list(output_shape), [10, 20, 1, 1])

    def test_as_3d(self):
        from inferno.extensions.layers.reshape import As3D
        from inferno.utils.exceptions import ShapeError

        as_3d = As3D()

        output_shape = as_3d(self._get_input_variable(10, 20, 3, 30, 30)).size()
        self.assertEqual(list(output_shape), [10, 20, 3, 30, 30])

        output_shape = as_3d(self._get_input_variable(10, 20, 30, 30)).size()
        self.assertEqual(list(output_shape), [10, 20, 1, 30, 30])

        output_shape = as_3d(self._get_input_variable(10, 20)).size()
        self.assertEqual(list(output_shape), [10, 20, 1, 1, 1])

        as_3d.channel_as_z = True
        output_shape = as_3d(self._get_input_variable(10, 20, 30, 30)).size()
        self.assertEqual(list(output_shape), [10, 1, 20, 30, 30])

        as_3d.num_channels_or_num_z_slices = 2
        output_shape = as_3d(self._get_input_variable(10, 40, 30, 30)).size()
        self.assertEqual(list(output_shape), [10, 2, 20, 30, 30])

        with self.assertRaises(ShapeError):
            output_shape = as_3d(self._get_input_variable(10, 41, 30, 30)).size()
            self.assertEqual(list(output_shape), [10, 2, 20, 30, 30])


if __name__ == '__main__':
    unittest.main()
