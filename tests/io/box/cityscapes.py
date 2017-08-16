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

    def test_cityscapes_dataset_with_transforms(self):
        from inferno.io.box.cityscapes import get_cityscapes_loader
        from inferno.utils.io_utils import print_tensor

        train_loader, validate_loader = get_cityscapes_loader(self.get_cityscapes_root())
        train_dataset = train_loader.dataset
        image, label = train_dataset[0]
        # Make sure the shapes checkout
        self.assertSequenceEqual(image.size(), (3, 360, 480))
        self.assertSequenceEqual(label.size(), (360, 480))
        self.assertEqual(image.type(), 'torch.FloatTensor')
        self.assertEqual(label.type(), 'torch.LongTensor')
        # Print tensors to make sure they look legit
        if not exists(self.PLOT_DIRECTORY):
            os.mkdir(self.PLOT_DIRECTORY)
        else:
            assert isdir(self.PLOT_DIRECTORY)
        print_tensor(image.numpy()[None, ...], prefix='IMG--', directory=self.PLOT_DIRECTORY)
        print_tensor(label.numpy()[None, None, ...], prefix='LAB--', directory=self.PLOT_DIRECTORY)
        print("[+] Inspect images at {}".format(self.PLOT_DIRECTORY))

if __name__ == '__main__':
    tester = TestCityscapes()
    tester.CITYSCAPES_ROOT = '/export/home/nrahaman/Python/Repositories/SegNet-Tutorial/CitYscapes'
    tester.test_cityscapes_dataset_without_transforms()