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

    def __init__(self, transform, accumulator='mean', close_channels=None, retain_segmentation=False):
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
        self.retain_segmentation = retain_segmentation

    def __call__(self, input_):
        transformed = np.array([self.transform(inp) for inp in input_])

        transformed = self.accumulator(transformed, axis=0)
        # t0 = np.mean(transformed, axis=0)
        # t1 = np.amax(transformed, axis=0)
        # t2 = np.amin(transformed, axis=0)

        if self.close_channels is not None:
            for c in self.close_channels:
                # TODO figure out what exactly size does
                transformed[c] = grey_opening(transformed[c], size=(3, 3))

        # if `retain_segmentation` is set to true, we just retain the 0th segmentation
        # to be compatible with the isbi loss functions
        if self.retain_segmentation:
            transformed = np.concatenate([input_[:1], transformed], axis=0)

        return transformed


def get_label_transforms(offsets, accumulator='mean', close_channels=None):
    from neurofire.transforms.segmentation import Segmentation2AffinitiesFromOffsets
    seg2aff = Segmentation2AffinitiesFromOffsets(dim=2,
                                                 offsets=pyu.from_iterable(offsets),
                                                 add_singleton_channel_dimension=True,
                                                 retain_segmentation=False)
    return AccumulateTransformOverLabelers(seg2aff, accumulator, close_channels, retain_segmentation=True)


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


def get_bsd500_loader(root_folder,
                      split,
                      offsets,
                      close_channels=None,
                      shuffle=True,
                      for_prediction=False,
                      accumulator='mean'):
    label_transforms = None if for_prediction else get_label_transforms(offsets,
                                                                        close_channels=close_channels,
                                                                        accumulator=accumulator)
    joint_transforms = None if for_prediction else get_joint_transforms()
    image_transforms = None if for_prediction else get_image_transforms()

    data_set = BSD500(root_folder,
                      subject='all',
                      split=split,
                      label_transform=label_transforms,
                      joint_transform=joint_transforms,
                      image_transform=image_transforms,
                      load_labels=not for_prediction)
    if for_prediction:
        return data_set
    else:
        return DataLoader(data_set, shuffle=shuffle)


class BSD500(Dataset):
    subject_modes = ('all',)
    splits = ('train', 'val', 'test')

    def __init__(self,
                 root_folder,
                 subject=None,
                 split='train',
                 image_transform=None,
                 joint_transform=None,
                 label_transform=None,
                 load_labels=True):
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
        if not os.path.exists(os.path.join(root_folder, "BSD500_%s.h5" % split)):
            print("creating h5 files of BSD500")

            import scipy.io as sio
            from scipy.ndimage import imread
            from glob import glob

            with h5py.File(os.path.join(root_folder, "BSD500_%s.h5" % split), "w") as out:
                for ds in ["train", "val", "test"]:
                    for i, f in enumerate(glob(root_folder + "/groundTruth/" + ds + "/*.mat")):
                        im_path = f.replace("groundTruth", "images").replace(".mat", ".jpg")
                        img = imread(im_path).transpose(2, 0, 1)
                        out.create_dataset("{}/image_data/{}".format(ds, i),
                                           data=img)
                        all_segmentations = sio.loadmat(f)["groundTruth"][0]
                        for subject in range(all_segmentations.shape[0]):
                            label_img = all_segmentations[subject][0][0][0]
                            out.create_dataset("{}/label_data/{}_{}".format(ds, i, subject),
                                               data=label_img)

        self.subject = subject
        self.root_folder = root_folder
        self.split = split

        self.image_transform = image_transform
        self.joint_transform = joint_transform
        self.label_transform = label_transform
        self.load_labels = load_labels

        self.load_data()

    def load_data(self):
        self.data = []
        with h5py.File(os.path.join(self.root_folder, "BSD500_%s.h5" % self.split), "r") as bsd:
            base = "{}/image_data/".format(self.split)
            label_base = base.replace("image_data", "label_data")

            img_nums = [int(num) for num in bsd["{}/image_data/".format(self.split)]]
            img_nums.sort()
            for img_num in img_nums:

                # load the image
                img_path = base + str(img_num)
                img = bsd[img_path].value.astype(np.float32)[:, 1:, 1:]

                # load the groundtruths
                subject_list = [p for p in bsd[label_base] if p.startswith("{}_".format(img_num))]
                if self.subject is None:
                    subject = choice(subject_list)
                elif isinstance(self.subject, int):
                    assert self.subject < len(subject_list)
                    subject = "{}_{}".format(img_num, self.subject)
                elif self.subject == 'all':
                    subject = subject_list

                label_path = "{}/{}".format(label_base, subject) if isinstance(subject, int) else \
                    ["{}/{}".format(label_base, subject_id) for subject_id in subject]

                gt = bsd[label_path].value.astype(np.float32)[1:, 1:] if isinstance(label_path, str) else \
                    np.array([bsd[lpath].value.astype(np.float32)[1:, 1:] for lpath in label_path])

                self.data.append((img, gt))

    def __getitem__(self, index):
        if index > len(self.data):
            index -= len(self.data)
        img, gt = self.data[index]

        if self.image_transform is not None:
            img = self.image_transform(img)

        if self.joint_transform is not None:
            img, gt = self.joint_transform(img, gt)

        if self.label_transform is not None:
            gt = self.label_transform(gt)

        if self.load_labels:
            return img, gt
        else:
            return img

    def __len__(self):
        return len(self.data)
