import os
from os.path import join, dirname, exists, isdir
import unittest
import numpy as np


_CAMVID_ROOT = None


def _camvid_available():
    return _CAMVID_ROOT is not None or os.environ.get('CAMVID_ROOT') is not None


class TestCamvid(unittest.TestCase):
    CAMVID_ROOT = _CAMVID_ROOT
    PLOT_DIRECTORY = join(dirname(__file__), 'plots')

    def get_camvid_root(self):
        if self.CAMVID_ROOT is None:
            root = os.environ.get('CAMVID_ROOT')
            assert root is not None, "Camvid Root not found."
        else:
            return self.CAMVID_ROOT

    @unittest.skipUnless(_camvid_available(), "No root available.")
    def test_camvid_dataset_without_transforms(self):
        from inferno.io.box.camvid import CamVid
        camvid = CamVid(self.get_camvid_root())
        image, label = camvid[0]
        image = np.asarray(image)
        label = np.asarray(label)
        self.assertSequenceEqual(image.shape, (360, 480, 3))
        self.assertSequenceEqual(label.shape, (360, 480))
        self.assertLessEqual(label.max(), 11)

    @unittest.skipUnless(_camvid_available(), "No root available.")
    def _test_camvid_dataset_with_transforms(self):
        from inferno.io.box.camvid import CamVid
        from inferno.io.transform.base import Compose
        from inferno.io.transform.image import PILImage2NumPyArray, RandomSizedCrop, Scale
        from inferno.utils.io_utils import print_tensor

        camvid = CamVid(self.get_camvid_root(),
                        image_transform=Compose(),
                        label_transform=Compose(),
                        joint_transform=Compose())
        camvid.image_transform.add(PILImage2NumPyArray())
        camvid.label_transform.add(PILImage2NumPyArray())
        image, label = camvid[0]
        self.assertSequenceEqual(image.shape, (3, 360, 480))
        self.assertSequenceEqual(label.shape, (360, 480))
        # Add crop trafo
        camvid.joint_transform.add(RandomSizedCrop(ratio_between=(0.7, 1.0),
                                                   preserve_aspect_ratio=True))
        # We need 2 scale transforms, one with order 3 (image) and the other with order 0 (label)
        camvid.joint_transform.add(Scale(output_image_shape=(360, 480),
                                         interpolation_order=3, apply_to=[0]))
        camvid.joint_transform.add(Scale(output_image_shape=(360, 480),
                                         interpolation_order=0, apply_to=[1]))
        image, label = camvid[0]
        self.assertSequenceEqual(image.shape, (3, 360, 480))
        self.assertSequenceEqual(label.shape, (360, 480))
        self.assertLessEqual(len(np.unique(label)), 12)
        # Print tensors to make sure they look legit
        if not exists(self.PLOT_DIRECTORY):
            os.mkdir(self.PLOT_DIRECTORY)
        else:
            assert isdir(self.PLOT_DIRECTORY)
        print_tensor(image[None, ...], prefix='IMG--', directory=self.PLOT_DIRECTORY)
        print_tensor(label[None, None, ...], prefix='LAB--', directory=self.PLOT_DIRECTORY)
        print("[+] Inspect images at {}".format(self.PLOT_DIRECTORY))

    @unittest.skipUnless(_camvid_available(), "No root available.")
    def test_camvid_dataset_with_transforms(self):
        from inferno.io.box.camvid import get_camvid_loaders
        from inferno.utils.io_utils import print_tensor

        train_loader, validate_loader, test_loader = get_camvid_loaders(self.get_camvid_root())
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

    @unittest.skipUnless(_camvid_available(), "No root available.")
    def test_camvid_dataset_with_transforms_onehot(self):
        from inferno.io.box.camvid import get_camvid_loaders
        from inferno.utils.io_utils import print_tensor

        train_loader, validate_loader, test_loader = get_camvid_loaders(self.get_camvid_root(),
                                                                        labels_as_onehot=True)
        train_dataset = train_loader.dataset
        image, label = train_dataset[0]
        # Make sure the shapes checkout
        self.assertSequenceEqual(image.size(), (3, 360, 480))
        self.assertSequenceEqual(label.size(), (12, 360, 480))
        self.assertEqual(image.type(), 'torch.FloatTensor')
        self.assertEqual(label.type(), 'torch.FloatTensor')
        # Print tensors to make sure they look legit
        if not exists(self.PLOT_DIRECTORY):
            os.mkdir(self.PLOT_DIRECTORY)
        else:
            assert isdir(self.PLOT_DIRECTORY)
        print_tensor(image.numpy()[None, ...], prefix='IMG--', directory=self.PLOT_DIRECTORY)
        print_tensor(label.numpy()[None, ...], prefix='LAB--', directory=self.PLOT_DIRECTORY)
        print("[+] Inspect images at {}".format(self.PLOT_DIRECTORY))


if __name__ == '__main__':
    unittest.main()
