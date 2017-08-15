import os
from os.path import join, dirname, exists, isdir
import unittest
import numpy as np


class TestCityscapes(unittest.TestCase):
    CITYSCAPES_ROOT = None
    PLOT_DIRECTORY = join(dirname(__file__), 'plots')

    def get_cityscapes_root(self):
        if self.CITYSCAPES_ROOT is None:
            root = os.environ.get('CITYSCAPES_ROOT')
            assert root is not None, "Cityscapes Root not found."
        else:
            return self.CITYSCAPES_ROOT

    def test_cityscapes_dataset_without_transforms(self):
        from inferno.io.box.cityscapes import Cityscapes
        cityscapes = Cityscapes(self.get_cityscapes_root())
        image, label = cityscapes[0]
        image = np.asarray(image)
        label = np.asarray(label)
        self.assertSequenceEqual(image.shape, (1024, 2048, 3))
        self.assertSequenceEqual(label.shape, (1024, 2048))
        self.assertLessEqual(label.max(), 33)

if __name__ == '__main__':
    tester = TestCityscapes()
    tester.CITYSCAPES_ROOT = '/export/home/nrahaman/Python/Repositories/SegNet-Tutorial/CitYscapes'
    tester.test_cityscapes_dataset_without_transforms()