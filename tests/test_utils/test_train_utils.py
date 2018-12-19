import unittest
import inferno.utils.train_utils as tu
import numpy as np


class FrequencyTest(unittest.TestCase):
    def test_from_string(self):
        frequency = tu.Frequency.from_string('10 epochs')
        self.assertFalse(frequency.match(epoch_count=9))
        self.assertTrue(frequency.match(epoch_count=10))
        frequency = tu.Frequency.from_string('1 iteration')
        self.assertEqual(frequency.units, 'iterations')
        self.assertTrue(frequency.match(iteration_count=10))
        frequency = tu.Frequency.from_string('never')
        self.assertFalse(frequency.match(epoch_count=9))
        frequency = tu.Frequency.from_string('inf epochs')
        self.assertFalse(frequency.match(epoch_count=9))

    def test_from_tuple(self):
        frequency = tu.Frequency.build_from((np.inf, 'epoch'))
        self.assertFalse(frequency.match(epoch_count=9))
        self.assertFalse(frequency.match(epoch_count=10))

    def test_is_consistent(self):
        frequency = tu.Frequency.build_from('10 epochs')
        frequency._units = 'banana'
        self.assertFalse(frequency.is_consistent)

    def test_init(self):
        frequency = tu.Frequency()
        self.assertEqual(frequency.value, np.inf)
        self.assertEqual(frequency.units, frequency.UNIT_PRIORITY)

    def test_duration(self):
        duration = tu.Duration.build_from((3, 'iterations'))
        self.assertFalse(duration.match(iteration_count=2))
        self.assertFalse(duration.match(iteration_count=3))
        self.assertTrue(duration.match(iteration_count=3, when_equal_return=True))
        self.assertTrue(duration.match(iteration_count=4))
        self.assertEqual(duration.compare(iteration_count=1, epoch_count=3).get('iterations'),
                         2)
        with self.assertRaises(ValueError):
            duration.match(epoch_count=2)


if __name__ == '__main__':
    unittest.main()