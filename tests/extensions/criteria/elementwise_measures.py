import unittest
import inferno.extensions.criteria.elementwise_measures as em
import torch
from torch.autograd import Variable


class TestElementwiseMeasures(unittest.TestCase):
    def test_weighted_mse_loss(self):
        input = Variable(torch.zeros(10, 10))
        target = Variable(torch.ones(10, 10))
        loss = em.WeightedMSELoss(positive_class_weight=2.)(input, target)
        self.assertAlmostEqual(loss.data[0], 2., delta=1e-5)
        target = Variable(torch.zeros(10, 10))
        input = Variable(torch.ones(10, 10))
        loss = em.WeightedMSELoss(positive_class_weight=2.)(input, target)
        self.assertAlmostEqual(loss.data[0], 1., delta=1e-5)


if __name__ == '__main__':
    unittest.main()