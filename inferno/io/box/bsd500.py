import numpy as np
from inferno.utils import python_utils as pyu
# import zipfile
import os  # , io
# from PIL import Image
# from scipy import ndimage
import h5py
from torch.utils.data.dataset import Dataset
from random import choice
from inferno.io.transform import Compose

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


class MeanTransformOverLabelers(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, input_):
        transformed = np.array([self.transform(inp) for inp in input_])
        return np.mean(transformed, axis=0)


def get_label_transforms(offsets):
    from neurofire.transforms.segmentation import Segmentation2AffinitiesFromOffsets
    seg2aff = Segmentation2AffinitiesFromOffsets(dim=2,
                                                 offsets=pyu.from_iterable(offsets),
                                                 add_singleton_channel_dimension=True,
                                                 retain_segmentation=False)
    return MeanTransformOverLabelers(seg2aff)


# TODO rotations, transpose, flips
def get_joint_transforms(offsets):
    pass


def get_bsd500_loader(root_folder, split, offsets):
    label_trafo = get_label_transforms(offsets)
    return BSD500(root_folder,
                  subject='all',
                  split=split,
                  label_transform=label_trafo)


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

        # FIXME this can be done by the dataloader ?!
        self.shuffle_data_paths()

    def shuffle_data_paths(self):
        self.data = []
        with h5py.File(os.path.join(self.root_folder, "BSD500.h5"), "r") as bsd:
            base = "{}/image_data/".format(self.split)
            label_base = base.replace("image_data", "label_data")

            for img_num in bsd["{}/image_data/".format(self.split)]:

                # load the image
                img_path = base + img_num
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

                # TODO apply all trafos
                gt = bsd[label_path].value.astype(np.float32)[1:, 1:] if isinstance(label_path, str) else \
                    np.array([bsd[lpath].value.astype(np.float32)[1:, 1:] for lpath in label_path])

                self.data.append((img, gt))

    def __getitem__(self, index):
        if index == 0 and self.subject is None:
            # TODO: shuffeling should actually be done, when a new batch is loaded
            self.shuffle_data_paths()
        if index > len(self.data):
            index -= len(self.data)
        img, gt = self.data[index]

        # TODO also apply image and joint transform
        if self.label_transform is not None:
            gt = self.label_transform(gt)
        return img, gt

    def __len__(self):
        return len(self.data)
