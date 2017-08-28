import unittest
import torch
from torch.autograd import Variable


class TestSorensenDice(unittest.TestCase):
    def get_dummy_variables(self):
        x = Variable(torch.zeros(3, 2, 100, 100).uniform_())
        y = Variable(torch.zeros(3, 2, 100, 100).uniform_())
        return x, y

    # noinspection PyCallingNonCallable
    def test_channelwise(self):
        from inferno.extensions.criteria.set_similarity_measures import SorensenDiceLoss
        x, y = self.get_dummy_variables()
        channelwise = SorensenDiceLoss(channelwise=True)
        not_channelwise = SorensenDiceLoss(channelwise=False)
        # Compute expected channelwise loss
        expected_channelwise_loss = \
            not_channelwise(x[:, 0, ...], y[:, 0, ...]) + \
            not_channelwise(x[:, 1, ...], y[:, 1, ...])
        # Compute channelwise
        channelwise_loss = channelwise(x, y)
        # Compare
        self.assertAlmostEqual(expected_channelwise_loss.data[0], channelwise_loss.data[0])


if __name__ == '__main__':
    unittest.main()
