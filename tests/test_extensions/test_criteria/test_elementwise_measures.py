import unittest
import inferno.extensions.criteria.elementwise_measures as em
import torch


class TestElementwiseMeasures(unittest.TestCase):
    def test_weighted_mse_loss(self):
        input = torch.zeros(10, 10)
        target = torch.ones(10, 10)
        loss = em.WeightedMSELoss(positive_class_weight=2.)(input, target)
        self.assertAlmostEqual(loss.item(), 2., delta=1e-5)
        target = torch.zeros(10, 10)
        input = torch.ones(10, 10)
        loss = em.WeightedMSELoss(positive_class_weight=2.)(input, target)
        self.assertAlmostEqual(loss.item(), 1., delta=1e-5)


if __name__ == '__main__':
    unittest.main()
