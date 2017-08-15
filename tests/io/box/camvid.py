import os
import unittest
import numpy as np


class TestCamvid(unittest.TestCase):
    CAMVID_ROOT = None

    def get_camvid_root(self):
        if self.CAMVID_ROOT is None:
            root = os.environ.get('CAMVID_ROOT')
            assert root is not None, "Camvid Root not found."
        else:
            return self.CAMVID_ROOT

    def test_camvid_dataset_without_transforms(self):
        from inferno.io.box.camvid import CamVid
        camvid = CamVid(self.get_camvid_root())
        image, label = camvid[0]
        image = np.asarray(image)
        label = np.asarray(label)
        self.assertSequenceEqual(image.shape, (360, 480, 3))
        self.assertSequenceEqual(label.shape, (360, 480))
        self.assertLessEqual(label.max(), 11)

    def test_camvid_dataset_with_transforms(self):
        from inferno.io.box.camvid import CamVid
        from inferno.io.transform.base import Compose
        from inferno.io.transform.image import PILImage2NumPyArray
        camvid = CamVid(self.get_camvid_root(),
                        image_transform=Compose(),
                        label_transform=Compose(),
                        joint_transform=Compose())
        camvid.image_transform.add(PILImage2NumPyArray())
        camvid.label_transform.add(PILImage2NumPyArray())
        image, label = camvid[0]
        self.assertSequenceEqual(image.shape, (3, 360, 480))
        self.assertSequenceEqual(label.shape, (360, 480))

if __name__ == '__main__':
    tester = TestCamvid()
    tester.CAMVID_ROOT = '/export/home/nrahaman/Python/Repositories/SegNet-Tutorial/CamVid'
    tester.test_camvid_dataset_with_transforms()