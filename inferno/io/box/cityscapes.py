import zipfile
import io
import os
import torch.utils.data as data
from PIL import Image
from os.path import join, relpath, abspath
from ...utils.exceptions import assert_
from ..transform.base import Compose
from ..transform.generic import \
    Normalize, NormalizeRange, Cast, AsTorchBatch, Project, Label2OneHot
from ..transform.image import \
    RandomSizedCrop, RandomGammaCorrection, RandomFlip, Scale, PILImage2NumPyArray
from ..core import Concatenate


CITYSCAPES_CLASSES = {
    0: 'unlabeled',
    1: 'ego vehicle',
    2: 'rectification border',
    3: 'out of roi',
    4: 'static',
    5: 'dynamic',
    6: 'ground',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
    -1: 'license plate'
}

IGNORE_CLASS_LABEL = 19

# Class labels to use for training, found here:
# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L61
CITYSCAPES_CLASSES_TO_LABELS = {
    0: IGNORE_CLASS_LABEL,
    1: IGNORE_CLASS_LABEL,
    2: IGNORE_CLASS_LABEL,
    3: IGNORE_CLASS_LABEL,
    4: IGNORE_CLASS_LABEL,
    5: IGNORE_CLASS_LABEL,
    6: IGNORE_CLASS_LABEL,
    7: 0,
    8: 1,
    9: IGNORE_CLASS_LABEL,
    10: IGNORE_CLASS_LABEL,
    11: 2,
    12: 3,
    13: 4,
    14: IGNORE_CLASS_LABEL,
    15: IGNORE_CLASS_LABEL,
    16: IGNORE_CLASS_LABEL,
    17: 5,
    18: IGNORE_CLASS_LABEL,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    29: IGNORE_CLASS_LABEL,
    30: IGNORE_CLASS_LABEL,
    31: 16,
    32: 17,
    33: 18,
    -1: IGNORE_CLASS_LABEL
}

# Map classes to official cityscapes colors
CITYSCAPES_CLASS_COLOR_MAPPING = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
    -1: (0, 0, 142),
}

# Weights corresponding to the outputs
CITYSCAPES_LABEL_WEIGHTS = {
    0: 1.,
    1: 1.,
    2: 1.,
    3: 1.,
    4: 1.,
    5: 1.,
    6: 1.,
    7: 1.,
    8: 1.,
    9: 1.,
    10: 1.,
    11: 1.,
    12: 1.,
    13: 1.,
    14: 1.,
    15: 1.,
    16: 1.,
    17: 1.,
    18: 1.,
    19: 0.
}

# 0:void 1:flat  2:construction  3:object  4:nature  5:sky  6:human  7:vehicle
CITYSCAPES_CATEGORIES = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
                         3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7]

CITYSCAPES_IGNORE_IN_EVAL = [True, True, True, True, True, True, True, False, False, True, True,
                             False, False, False, True, True, True, False, True, False, False,
                             False, False, False, False,
                             False, False, False, False, True, True, False, False, False, True]

# mean and std
CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]


def get_matching_labelimage_file(f, groundtruth):
    fs = f.split('/')
    fs[0] = groundtruth
    fs[-1] = str.replace(fs[-1], 'leftImg8bit', groundtruth + '_labelIds')
    return '/'.join(fs)


def get_filelist(path):
    if path.endswith('.zip'):
        return zipfile.ZipFile(path, 'r').filelist
    elif os.path.isdir(path):
        return [relpath(join(root, filename), abspath(join(path, '..')))
                for root, _, filenames in os.walk(path) for filename in filenames]
    else:
        raise NotImplementedError("Path must be a zip archive or a directory.")


def make_dataset(path, split):
    images = []
    for f in get_filelist(path):
        if isinstance(f, str):
            fn = f
            fns = f.split('/')
        else:
            fn = f.filename
            fns = f.filename.split('/')
        if fns[-1].endswith('.png') and fns[1] == split:
            # use first folder name to identify train/val/test images
            if split == 'train_extra':
                groundtruth = 'gtCoarse'
            else:
                groundtruth = 'gtFine'

            fl = get_matching_labelimage_file(fn, groundtruth)
            images.append((f, fl))
    return images


def extract_image(path, image_path):
    if path.endswith('.zip'):
        # read image directly from zipfile if path is a zip
        return Image.open(io.BytesIO(zipfile.ZipFile(path, 'r').read(image_path)))
    else:
        return Image.open(join(abspath(join(path, '..')), image_path), 'r')


class Cityscapes(data.Dataset):
    SPLIT_NAME_MAPPING = {'train': 'train',
                          'training': 'train',
                          'validate': 'val',
                          'val': 'val',
                          'validation': 'val',
                          'test': 'test',
                          'testing': 'test',
                          'training_extra': 'train_extra',
                          'train_extra': 'train_extra'}

    # Dataset statistics
    CLASSES = CITYSCAPES_CLASSES
    MEAN = CITYSCAPES_MEAN
    STD = CITYSCAPES_STD

    BLACKLIST = ['leftImg8bit/train_extra/troisdorf/troisdorf_000000_000073_leftImg8bit.png']

    def __init__(self, root_folder, split='train', read_from_zip_archive=True,
                 image_transform=None, label_transform=None, joint_transform=None):
        """
        Parameters:
        root_folder: folder that contains both leftImg8bit_trainvaltest.zip and
               gtFine_trainvaltest.zip archives.
        split: name of dataset spilt (i.e. 'train_extra', 'train', 'val' or 'test') 
        """

        assert_(split in self.SPLIT_NAME_MAPPING.keys(),
                "`split` must be one of {}".format(set(self.SPLIT_NAME_MAPPING.keys())),
                KeyError)
        self.split = self.SPLIT_NAME_MAPPING.get(split)
        self.read_from_zip_archive = read_from_zip_archive

        # Get roots
        self.image_root, self.label_root = [join(root_folder, groot)
                                            for groot in self.get_image_and_label_roots()]

        # Transforms
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        # Make list with paths to the images
        self.image_paths = make_dataset(self.image_root, self.split)

    def __getitem__(self, index):
        pi, pl = self.image_paths[index]
        if pi in self.BLACKLIST:
            # Select the next image if the current image is bad
            return self[index + 1]
        image = extract_image(self.image_root, pi)
        label = extract_image(self.label_root, pl)
        try:
            # Apply transforms
            if self.image_transform is not None:
                image = self.image_transform(image)
            if self.label_transform is not None:
                label = self.label_transform(label)
            if self.joint_transform is not None:
                image, label = self.joint_transform(image, label)
        except Exception:
            print("[!] An Exception occurred while applying the transforms at "
                  "index {} of split '{}'.".format(index, self.split))
            raise
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def download(self):
        # TODO: please download the dataset from
        # https://www.cityscapes-dataset.com/
        raise NotImplementedError

    def get_image_and_label_roots(self):
        all_roots = {
            'zipped':
                {
                    'train': ('leftImg8bit_trainvaltest.zip', 'gtFine_trainvaltest.zip'),
                    'val': ('leftImg8bit_trainvaltest.zip', 'gtFine_trainvaltest.zip'),
                    'train_extra': ('leftImg8bit_trainextra.zip', 'gtCoarse.zip')
                },
            'unzipped':
                {
                    'train': ('leftImg8bit', 'gtFine'),
                    'val': ('leftImg8bit', 'gtFine'),
                    'train_extra': ('leftImg8bit', 'gtCoarse')
                }
        }
        image_and_label_roots = all_roots\
            .get('zipped' if self.read_from_zip_archive else 'unzipped').get(self.split)
        return image_and_label_roots


def make_transforms(image_shape, labels_as_onehot):
    # Make transforms
    image_transforms = Compose(PILImage2NumPyArray(),
                               NormalizeRange(),
                               RandomGammaCorrection(),
                               Normalize(mean=CITYSCAPES_MEAN, std=CITYSCAPES_STD))
    label_transforms = Compose(PILImage2NumPyArray(),
                               Project(projection=CITYSCAPES_CLASSES_TO_LABELS))
    joint_transforms = Compose(RandomSizedCrop(ratio_between=(0.6, 1.0),
                                               preserve_aspect_ratio=True),
                               # Scale raw image back to the original shape
                               Scale(output_image_shape=image_shape,
                                     interpolation_order=3, apply_to=[0]),
                               # Scale segmentation back to the original shape
                               # (without interpolation)
                               Scale(output_image_shape=image_shape,
                                     interpolation_order=0, apply_to=[1]),
                               RandomFlip(allow_ud_flips=False),
                               # Cast raw image to float
                               Cast('float', apply_to=[0]))
    if labels_as_onehot:
        # Applying Label2OneHot on the full label image makes it unnecessarily expensive,
        # because we're throwing it away with RandomSizedCrop and Scale. Tests show that it's
        # ~1 sec faster per image.
        joint_transforms \
            .add(Label2OneHot(num_classes=len(CITYSCAPES_LABEL_WEIGHTS), dtype='bool',
                              apply_to=[1])) \
            .add(Cast('float', apply_to=[1]))
    else:
        # Cast label image to long
        joint_transforms.add(Cast('long', apply_to=[1]))
    # Batchify
    joint_transforms.add(AsTorchBatch(2, add_channel_axis_if_necessary=False))
    # Return as kwargs
    return {'image_transform': image_transforms,
            'label_transform': label_transforms,
            'joint_transform': joint_transforms}


def get_cityscapes_loaders(root_directory, image_shape=(1024, 2048), labels_as_onehot=False,
                           include_coarse_dataset=False, read_from_zip_archive=True,
                           train_batch_size=1, validate_batch_size=1, num_workers=2):
    # Build datasets
    train_dataset = Cityscapes(root_directory, split='train',
                               read_from_zip_archive=read_from_zip_archive,
                               **make_transforms(image_shape, labels_as_onehot))
    if include_coarse_dataset:
        # Build coarse dataset
        coarse_dataset = Cityscapes(root_directory, split='train_extra',
                                    read_from_zip_archive=read_from_zip_archive,
                                    **make_transforms(image_shape, labels_as_onehot))
        # ... and concatenate with train_dataset
        train_dataset = Concatenate(coarse_dataset, train_dataset)
    validate_dataset = Cityscapes(root_directory, split='validate',
                                  read_from_zip_archive=read_from_zip_archive,
                                  **make_transforms(image_shape, labels_as_onehot))

    # Build loaders
    train_loader = data.DataLoader(train_dataset, batch_size=train_batch_size,
                                   shuffle=True, num_workers=num_workers, pin_memory=True)
    validate_loader = data.DataLoader(validate_dataset, batch_size=validate_batch_size,
                                      shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader, validate_loader
