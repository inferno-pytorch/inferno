import numpy as np
import zipfile
import os, io
from PIL import Image
from scipy import ndimage
from os import listdir
from os.path import join, exists
import h5py
from torch.utils.data.dataset import Dataset
from random import choice

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


class BSD500(Dataset):
    def __init__(self, root_folder, subject=None, split='train'):
        """
        Parameters:
        root_folder: folder that contains 'groundTruth' and 'images'  of the BSD 500.
        (http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)
        """
        self.root_folder = root_folder
        if not exists(join(root_folder, "BSD500.h5")):
            print("creating h5 files of BSD500")
            
            import scipy.io as sio
            from scipy.ndimage import imread
            from glob import glob
            num_subjects = 1

            with h5py.File(join(root_folder, "BSD500.h5"), "w") as out:
                for ds in ["train", "val", "test"]:
                    for i, f in enumerate(glob(root_folder+"/groundTruth/"+ds+"/*.mat")):
                        im_path = f.replace("groundTruth", "images").replace(".mat", ".jpg")
                        img = imread(im_path).transpose(2, 0, 1)
                        image_shape = "x".join(str(x) for x in img.shape)
                        out.create_dataset("{}/{}/image_data/{}".format(ds, image_shape, i), data=img)
                        all_segmentations = sio.loadmat(f)["groundTruth"][0]
                        for subject in range(all_segmentations.shape[0]):
                            label_img = all_segmentations[subject][0][0][0]
                            out.create_dataset("{}/{}/label_data/{}_{}".format(ds, image_shape, i, subject), data=label_img)

        self.subject = subject
        self.root_folder = root_folder
        self.split = split

        self.shuffle_data_paths()


    def shuffle_data_paths(self):
        self.data = {}
        with h5py.File(join(self.root_folder, "BSD500.h5"), "r") as bsd:
            self.shapes = list(bsd[self.split].keys())

            for shape in self.shapes:
                self.data[shape] = []
                base = "{}/{}/image_data/".format(self.split, shape)
                label_base = base.replace("image_data", "label_data")
                
                for img_num in bsd["{}/{}/image_data/".format(self.split, shape)]:
                    img_path = base + img_num
                    if self.subject is None:
                        subject = choice([p for p in bsd[label_base] if p.startswith("{}_".format(img_num))])
                    else:
                        subject = "{}_{}".format(img_num, self.subject)

                    label_path = "{}/{}".format(label_base, subject)
                    self.data[shape].append((bsd[img_path].value.astype(np.float32)[:, 1:, 1:], 
                                             bsd[label_path].value.astype(np.float32)[1:, 1:]))


    def __getitem__(self, index):
        if index == 0 and self.subject is None:
            # TODO: shuffeling should actually be done, when a new batch is loaded
            self.shuffle_data_paths()
        for k in self.data:
            if index > len(self.data[k]):
                index -= len(self.data[k])
            else:
                return self.data[k][index]

    def __len__(self):
        return sum(len(self.data[k]) for k in self.data)
