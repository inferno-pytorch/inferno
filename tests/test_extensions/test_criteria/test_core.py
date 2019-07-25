import unittest
import torch
import torch.nn as nn


class TestCore(unittest.TestCase):
    def test_as_2d_criterion(self):
        from inferno.extensions.criteria.core import As2DCriterion

        prediction = torch.FloatTensor(2, 10, 100, 100).uniform_()
        prediction = nn.Softmax2d()(prediction)
        target = torch.LongTensor(2, 100, 100).fill_(0)
        criterion = As2DCriterion(nn.CrossEntropyLoss())
        criterion(prediction, target)


if __name__ == '__main__':
    unittest.main()
