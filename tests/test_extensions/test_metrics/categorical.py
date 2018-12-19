import unittest
import torch
from inferno.extensions.metrics import IOU


class TestCategorical(unittest.TestCase):
    def test_iou_basic(self):
        # from one hot
        predicted_image = torch.zeros(*(2, 10, 10))
        predicted_image[:, 0:4, 0:4] = 1
        target_image = torch.zeros(*(2, 10, 10))
        target_image[:, 0:3, 0:3] = 1
        expected_iou = (3 * 3)/(4 * 4)
        iou = IOU()(predicted_image[None, ...], target_image[None, ...])
        self.assertAlmostEqual(iou, expected_iou, places=4)

    def test_iou_with_ignore_class(self):
        predicted_image = torch.zeros(*(2, 10, 10))
        predicted_image[0, 0:4, 0:4] = 1
        target_image = torch.zeros(*(2, 10, 10))
        target_image[:, 0:3, 0:3] = 1
        expected_iou = (3 * 3) / (4 * 4)
        iou = IOU(ignore_class=1)(predicted_image[None, ...], target_image[None, ...])
        self.assertAlmostEqual(iou, expected_iou, places=4)

    def test_multiclass_iou(self):
        predicted_image = torch.zeros(*(2, 10, 10))
        predicted_image[0, 0:4, 0:4] = 1
        target_image = torch.zeros(*(2, 10, 10))
        target_image[:, 0:3, 0:3] = 1
        iou_class_0 = (3 * 3) / (4 * 4)
        iou_class_1 = 0
        expected_mean_iou = 0.5 * (iou_class_0 + iou_class_1)
        iou = IOU()(predicted_image[None, ...], target_image[None, ...])
        self.assertAlmostEqual(iou, expected_mean_iou, places=4)

    def test_multiclass_iou_with_ignore_class(self):
        predicted_image = torch.zeros(*(3, 10, 10))
        predicted_image[0, 0:4, 0:4] = 1
        # Have the third plane be crap
        predicted_image[2, :, :] = 1
        target_image = torch.zeros(*(3, 10, 10))
        target_image[:, 0:3, 0:3] = 1
        iou_class_0 = (3 * 3) / (4 * 4)
        iou_class_1 = 0
        expected_mean_iou = 0.5 * (iou_class_0 + iou_class_1)
        iou = IOU(ignore_class=-1)(predicted_image[None, ...], target_image[None, ...])
        self.assertAlmostEqual(iou, expected_mean_iou, places=4)

if __name__ == '__main__':
    unittest.main()