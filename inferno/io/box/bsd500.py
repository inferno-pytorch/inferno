import numpy as np
from inferno.utils import python_utils as pyu
# import zipfile
import os  # , io
# from PIL import Image
# from scipy import ndimage
import h5py
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from random import choice
from ..transform import Compose
from ..transform.generic import Normalize, NormalizeRange  # , Cast, AsTorchBatch
from scipy.ndimage import grey_opening

# def get_matching_labelimage_file(f):
#     fs = f.split('/')
#     fn[0] = "gtFine"
#     fn[-1] = str.replace(fn[-1], 'leftImg8bit', 'gtFine_labelIds')
#     return join(fn)

# def make_dataset(image_zip_file, split):
#     images = []
#     for f in zipfile.ZipFile(image_zip_file, 'r').filelist:
#         fn = f.filename.split('/')
#         if fn[-1].endswith('.png') and fn[1] == split:
#             # use first folder name to identify train/val/test images
#             fl = get_matching_labelimage_file(f)
#             images.append((f, fl))
#     return images

# def get_image(archive, image_path):
#     # read image directly from zipfile
#     return np.array(Image.open(io.BytesIO(zipfile.ZipFile(archive, 'r').read(image_path))))


class AccumulateTransformOverLabelers(object):
    accumulators = ('mean', 'max', 'min')

    def __init__(self, transform, accumulator='mean', close_channels=None):
        self.transform = transform
        assert accumulator in self.accumulators
        if accumulator == 'mean':
            self.accumulator = np.mean
        elif accumulator == 'max':
            self.accumulator = np.amax
        elif accumulator == 'min':
            self.accumulator = np.amin
        if close_channels is not None:
            assert isinstance(close_channels, (list, tuple))
        self.close_channels = close_channels

    def __call__(self, input_):
        transformed = np.array([self.transform(inp) for inp in input_])
        transformed = self.accumulator(transformed, axis=0)
        if self.close_channels is not None:
            for c in self.close_channels:
                # TODO figure out what exactly size does
                transformed[c] = grey_opening(transformed[c], size=(3, 3))
        return transformed


def get_label_transforms(offsets, accumulator='mean', close_channels=None):
    from neurofire.transforms.segmentation import Segmentation2AffinitiesFromOffsets
    seg2aff = Segmentation2AffinitiesFromOffsets(dim=2,
                                                 offsets=pyu.from_iterable(offsets),
                                                 add_singleton_channel_dimension=True,
                                                 retain_segmentation=False)
    return AccumulateTransformOverLabelers(seg2aff, accumulator, close_channels)


def get_joint_transforms():
    from ..transform.image import RandomFlip, RandomRotate, RandomTranspose
    trafos = Compose(RandomFlip(allow_ud_flips=False), RandomRotate(), RandomTranspose())
    return trafos


# TODO gamma correction ?
def get_image_transforms():
    trafos = Compose(NormalizeRange(),
                     # RandomGammaCorrection(),
                     Normalize())
    return trafos


# TODO return data loaders for train, val and test
def get_bsd500_loaders(root_folder, offsets, close_channels=None, shuffle=True):
    label_transforms = get_label_transforms(offsets, close_channels=close_channels)
    joint_transforms = get_joint_transforms()
    image_transforms = get_image_transforms()

    train_set = BSD500(root_folder,
                       subject='all',
                       split='train',
                       label_transform=label_transforms,
                       joint_transform=joint_transforms,
                       image_transform=image_transforms)
    val_set = BSD500(root_folder,
                     subject='all',
                     split='val',
                     label_transform=label_transforms,
                     joint_transform=joint_transforms,
                     image_transform=image_transforms)
    test_set = BSD500(root_folder,
                      subject='all',
                      split='test',
                      label_transform=label_transforms,
                      joint_transform=joint_transforms,
                      image_transform=image_transforms)

    return DataLoader(train_set, shuffle=shuffle), \
           DataLoader(val_set, shuffle=shuffle), \
           DataLoader(test_set, shuffle=shuffle)


class BSD500(Dataset):
    subject_modes = ('all',)
    splits = ('train', 'val', 'test')

    def __init__(self,
                 root_folder,
                 subject=None,
                 split='train',
                 image_transform=None,
                 joint_transform=None,
                 label_transform=None):
        """
        Parameters:
        root_folder: folder that contains 'groundTruth' and 'images'  of the BSD 500.
        (http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)
        subject: defines which labeler should be used / how to combine the labelers.
                 integer -> labeler with this number is drawn
                 None -> random labeler is drawn
                 'all' labelers are drawn (and potentially combined)
        """
        # validate
        assert os.path.exists(root_folder)
        if subject is not None:
            assert isinstance(subject, (int, str))
            if isinstance(subject, str):
                assert subject in self.subject_modes
        assert split in self.splits, str(split)

        self.root_folder = root_folder
        if not os.path.exists(os.path.join(root_folder, "BSD500.h5")):
            print("creating h5 files of BSD500")

            import scipy.io as sio
            from scipy.ndimage import imread
            from glob import glob

            with h5py.File(os.path.join(root_folder, "BSD500.h5"), "w") as out:
                for ds in ["train", "val", "test"]:
                    for i, f in enumerate(glob(root_folder + "/groundTruth/" + ds + "/*.mat")):
                        im_path = f.replace("groundTruth", "images").replace(".mat", ".jpg")
                        img = imread(im_path).transpose(2, 0, 1)
                        out.create_dataset("{}/image_data/{}".format(ds, i),
                                           data=img)
                        all_segmentations = sio.loadmat(f)["groundTruth"][0]
                        label_img = np.stack([s[0][0][0] for s in all_segmentations])
                        out.create_dataset("{}/label_data/{}".format(ds, i),
                                           data=label_img)

        self.subject = subject
        self.root_folder = root_folder
        self.split = split

        self.image_transform = image_transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform
        self.load_data()

    def load_data(self):
        self.data = []
        with h5py.File(os.path.join(self.root_folder, "BSD500.h5"), "r") as bsd:
            base = "{}/image_data/".format(self.split)
            label_base = base.replace("image_data", "label_data")

            for img_num in bsd["{}/image_data/".format(self.split)]:

                # load the image
                img_path = base + img_num
                img = bsd[img_path].value.astype(np.float32)[:, 1:, 1:]

                # load the groundtruths
                gt_path = label_base + img_num
                gt = bsd[gt_path].value.astype(np.float32)[1:, 1:]

                self.data.append((img, gt))

    def __getitem__(self, index):
        img, gt = self.data[index]

        if self.image_transform is not None:
            img = self.image_transform(img)

        if self.joint_transform is not None:
            img, gt = self.joint_transform(img, gt)

        if self.label_transform is not None:
            gt = self.label_transform(gt)
        return img, gt

    def __len__(self):
        return len(self.data)
