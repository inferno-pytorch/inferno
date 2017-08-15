import numpy as np
import zipfile
import os, io
from PIL import Image
from scipy import ndimage
from os import listdir
from os.path import join

CITYSCAPES_CLASSES = {
     0: 'unlabeled'            ,
     1: 'ego vehicle'          ,
     2: 'rectification border' ,
     3: 'out of roi'           ,
     4: 'static'               ,
     5: 'dynamic'              ,
     6: 'ground'               ,
     7: 'road'                 ,
     8: 'sidewalk'             ,
     9: 'parking'              ,
    10: 'rail track'           ,
    11: 'building'             ,
    12: 'wall'                 ,
    13: 'fence'                ,
    14: 'guard rail'           ,
    15: 'bridge'               ,
    16: 'tunnel'               ,
    17: 'pole'                 ,
    18: 'polegroup'            ,
    19: 'traffic light'        ,
    20: 'traffic sign'         ,
    21: 'vegetation'           ,
    22: 'terrain'              ,
    23: 'sky'                  ,
    24: 'person'               ,
    25: 'rider'                ,
    26: 'car'                  ,
    27: 'truck'                ,
    28: 'bus'                  ,
    29: 'caravan'              ,
    30: 'trailer'              ,
    31: 'train'                ,
    32: 'motorcycle'           ,
    33: 'bicycle'              ,
    -1: 'license plate'        
}

#0:void 1:flat  2:construction  3:object  4:nature  5:sky  6:human  7:vehicle
CITYSCAPES_CATEGORIES = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, \
                         3, 3, 3, 3, 4, 4, 5, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7]

CITYSCAPES_IGNORE_IN_EVAL = [True, True, True, True, True, True, True, False, False, True, True, \
    False, False, False, True, True, True, False, True, False, False, False, False, False, False,\
    False, False, False, False, True, True, False, False, False, True]

# mean and std
CITYSCAPES_MEAN = [0.28689554, 0.32513303, 0.28389177]
CITYSCAPES_STD = [0.18696375, 0.19017339, 0.18720214]


def get_matching_labelimage_file(f):
    fs = f.split('/')
    fn[0] = "gtFine"
    fn[-1] = str.replace(fn[-1], 'leftImg8bit', 'gtFine_labelIds')
    return join(fn)

def make_dataset(image_zip_file, split):
    images = []
    for f in zipfile.ZipFile(image_zip_file, 'r').filelist:
        fn = f.filename.split('/')
        if fn[-1].endswith('.png') and fn[1] == split:
            # use first folder name to identify train/val/test images
            fl = get_matching_labelimage_file(f)
            images.append((f, fl))
    return images

def get_image(archive, image_path):
    # read image directly from zipfile
    return np.array(Image.open(io.BytesIO(zipfile.ZipFile(archive, 'r').read(image_path))))


class Cityscapes(data.Dataset):
    SPLIT_NAME_MAPPING = {'train': 'train',
                          'training': 'train',
                          'validate': 'val',
                          'val': 'val',
                          'validation': 'val',
                          'test': 'test',
                          'testing': 'test'}
    # Dataset statistics
    CLASS_WEIGHTS = CITYSCAPES_CLASS_WEIGHTS
    CLASSES = CITYSCAPES_CLASSES
    MEAN = CITYSCAPES_MEAN
    STD = CITYSCAPES_STD

    def __init__(self, root_folder, split='train',
                 image_transform=None, label_transform=None, joint_transform=None):
        """
        Parameters:
        root_folder: folder that contains both leftImg8bit_trainvaltest.zip and
               gtFine_trainvaltest.zip archives.
        split: name of dataset spilt (i.e. 'train', 'val' or 'test') 
        """

        self.image_zip_file = join(root_folder,'/leftImg8bit_trainvaltest.zip')
        self.label_zip_file = join(root_folder,'/gtFine_trainvaltest.zip')
        assert_(split in self.SPLIT_NAME_MAPPING.keys(),
                "`split` must be one of {}".format(set(self.SPLIT_NAME_MAPPING.keys())),
                KeyError)
        self.split = self.SPLIT_NAME_MAPPING.get(split)
        # Transforms
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform
        # Make list with paths to the images
        self.image_paths = self.make_dataset(self.root_folder, self.split)

    def __getitem__(self, index):
        pi, pl = self.image_paths[index]
        image = self.get_image(self.image_zip_file, pi)
        label = self.get_image(self.label_zip_file, pl)
        # Apply transforms
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.label_transform is not None:
            label = self.label_transform(label)
        if self.joint_transform is not None:
            image, label = self.joint_transform(image, label)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def download(self):
        # TODO: please download the dataset from
        # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CitYscapes
        raise NotImplementedError
