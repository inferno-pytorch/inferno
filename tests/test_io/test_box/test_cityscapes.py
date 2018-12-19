import os
from os.path import join, dirname, exists, isdir
import unittest
import numpy as np
import time

_CITYSCAPES_ROOT = None


def _cityscapes_available():
    return _CITYSCAPES_ROOT is not None or os.environ.get('CITYSCAPES_ROOT') is not None


class TestCityscapes(unittest.TestCase):
    CITYSCAPES_ROOT = _CITYSCAPES_ROOT
    PLOT_DIRECTORY = join(dirname(__file__), 'plots')
    INCLUDE_COARSE = False

    def get_cityscapes_root(self):
        if self.CITYSCAPES_ROOT is None:
            root = os.environ.get('CITYSCAPES_ROOT')
            assert root is not None, "Cityscapes Root not found."
        else:
            return self.CITYSCAPES_ROOT

    @unittest.skipUnless(_cityscapes_available(), "No cityscapes available.")
    def test_cityscapes_dataset_without_transforms(self):
        from inferno.io.box.cityscapes import Cityscapes
        cityscapes = Cityscapes(self.get_cityscapes_root())
        image, label = cityscapes[0]
        image = np.asarray(image)
        label = np.asarray(label)
        self.assertSequenceEqual(image.shape, (1024, 2048, 3))
        self.assertSequenceEqual(label.shape, (1024, 2048))
        self.assertLessEqual(label.max(), 33)

    @unittest.skipUnless(_cityscapes_available(), "No cityscapes available.")
    def test_cityscapes_dataset_without_transforms_unzipped(self):
        from inferno.io.box.cityscapes import Cityscapes
        cityscapes = Cityscapes(join(self.get_cityscapes_root(), 'extracted'),
                                read_from_zip_archive=False)
        image, label = cityscapes[0]
        image = np.asarray(image)
        label = np.asarray(label)
        self.assertSequenceEqual(image.shape, (1024, 2048, 3))
        self.assertSequenceEqual(label.shape, (1024, 2048))
        self.assertLessEqual(label.max(), 33)

    @unittest.skipUnless(_cityscapes_available(), "No cityscapes available.")
    def test_cityscapes_dataset_with_transforms(self):
        from inferno.io.box.cityscapes import get_cityscapes_loaders
        from inferno.utils.io_utils import print_tensor

        train_loader, validate_loader = get_cityscapes_loaders(self.get_cityscapes_root(),
                                                               include_coarse_dataset=self.INCLUDE_COARSE)
        train_dataset = train_loader.dataset
        tic = time.time()
        image, label = train_dataset[0]
        toc = time.time()
        print("[+] Loaded sample in {} seconds.".format(toc - tic))
        # Make sure the shapes checkout
        self.assertSequenceEqual(image.size(), (3, 1024, 2048))
        self.assertSequenceEqual(label.size(), (1024, 2048))
        self.assertEqual(image.type(), 'torch.FloatTensor')
        self.assertEqual(label.type(), 'torch.LongTensor')
        # Print tensors to make sure they look legit
        if not exists(self.PLOT_DIRECTORY):
            os.mkdir(self.PLOT_DIRECTORY)
        else:
            assert isdir(self.PLOT_DIRECTORY)
        print_tensor(image.numpy()[None, ...], prefix='IMG--', directory=self.PLOT_DIRECTORY)
        for class_id in np.unique(label.numpy()):
            print_tensor((label.numpy()[None, None, ...] == class_id).astype('float32'),
                         prefix='LAB-{}--'.format(class_id),
                         directory=self.PLOT_DIRECTORY)
        print_tensor(label.numpy()[None, None, ...],
                     prefix='LAB--',
                     directory=self.PLOT_DIRECTORY)
        print("[+] Inspect images at {}".format(self.PLOT_DIRECTORY))

    @unittest.skipUnless(_cityscapes_available(), "No cityscapes available.")
    def test_cityscapes_dataset_with_transforms_unzipped(self):
        from inferno.io.box.cityscapes import get_cityscapes_loaders
        from inferno.utils.io_utils import print_tensor

        train_loader, validate_loader = get_cityscapes_loaders(join(self.get_cityscapes_root(),
                                                                    'extracted'),
                                                               include_coarse_dataset=self.INCLUDE_COARSE,
                                                               read_from_zip_archive=False)
        train_dataset = train_loader.dataset
        tic = time.time()
        image, label = train_dataset[0]
        toc = time.time()
        print("[+] Loaded sample in {} seconds.".format(toc - tic))
        # Make sure the shapes checkout
        self.assertSequenceEqual(image.size(), (3, 1024, 2048))
        self.assertSequenceEqual(label.size(), (1024, 2048))
        self.assertEqual(image.type(), 'torch.FloatTensor')
        self.assertEqual(label.type(), 'torch.LongTensor')
        # Print tensors to make sure they look legit
        if not exists(self.PLOT_DIRECTORY):
            os.mkdir(self.PLOT_DIRECTORY)
        else:
            assert isdir(self.PLOT_DIRECTORY)
        print_tensor(image.numpy()[None, ...], prefix='IMG--', directory=self.PLOT_DIRECTORY)
        for class_id in np.unique(label.numpy()):
            print_tensor((label.numpy()[None, None, ...] == class_id).astype('float32'),
                         prefix='LAB-{}--'.format(class_id),
                         directory=self.PLOT_DIRECTORY)
        print_tensor(label.numpy()[None, None, ...],
                     prefix='LAB--',
                     directory=self.PLOT_DIRECTORY)
        print("[+] Inspect images at {}".format(self.PLOT_DIRECTORY))


if __name__ == '__main__':
    unittest.main()
