import torch.utils.data as data
import skimage.data
import numpy
from operator import mul
from functools import reduce

class BinaryBlobs(data.Dataset):


    def __init__(self, size=20, length=512, blob_size_fraction=0.1,
                 n_dim=2, volume_fraction=0.5,split='train', 
                 uniform_noise_range=(-1.2, 1.2),
                 gaussian_noise_sigma=1.2,
                 noise_scale_factor=8,
                 image_transform=None, 
                 label_transform=None, 
                 joint_transform=None):
        # how many images are in the dataset
        self.size = size

        # blob related members
        self.length             = length
        self.blob_size_fraction = blob_size_fraction
        self.n_dim              = n_dim
        self.volume_fraction    = volume_fraction

        # which split {'train', 'test', 'validate'}
        self.split              = split

        # noise related members
        self.uniform_noise_range = uniform_noise_range
        self.gaussian_noise_sigma = float(gaussian_noise_sigma)
        self.noise_scale_factor = noise_scale_factor

        # transforms
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.joint_transform = joint_transform

        # internal
        split_to_seed = dict(train=0, test=1, validate=2)
        self.master_seed  = split_to_seed[self.split]*self.size

    def __getitem__(self, index):

        # generate the labels
        label = skimage.data.binary_blobs(
            length=self.length, 
            blob_size_fraction=self.blob_size_fraction, 
            n_dim=self.n_dim, 
            volume_fraction=self.volume_fraction,
            seed=self.master_seed + index)

        # make the raw image [-1,1]
        image  = label.astype('float32')*2
        image -= 1


        # add uniform noise 
        low, high = self.uniform_noise_range
        uniform_noise   = numpy.random.uniform(low=low, high=high, 
                                               size=image.size)
        image += uniform_noise.reshape(image.shape)

        # add gaussian noise
        gaussian_noise   = numpy.random.normal(scale=self.gaussian_noise_sigma, 
                                              size=image.size)
        image += gaussian_noise.reshape(image.shape)


        # generate noise at lower scales
        small_shape = [s//self.noise_scale_factor for s in label.shape]
        small_size = reduce(mul, small_shape, 1)
        small_noise_img   = numpy.random.uniform(low=low, high=high, 
                                               size=small_size)
        small_noise_img   = small_noise_img.reshape(small_shape)

        gaussian_noise   = numpy.random.normal(scale=self.gaussian_noise_sigma, 
                                              size=small_size)
        small_noise_img += gaussian_noise.reshape(small_shape)

        noise_img = skimage.transform.resize(image = small_noise_img, 
            output_shape=image.shape,  mode='reflect')


        image += noise_img

        image -= image.mean()
        image /= image.std()
        
        label = label.astype('long')
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

        image = image[None,...]
        return image, label

    def __len__(self):
        return self.size


def get_binary_blob_loaders(train_batch_size=1, test_batch_size=1,
                            num_workers=1,
                            train_image_transform=None,
                            train_label_transform=None,
                            train_joint_transform=None,
                            validate_image_transform=None,
                            validate_label_transform=None,
                            validate_joint_transform=None,
                            test_image_transform=None,
                            test_label_transform=None,
                            test_joint_transform=None,
                            **kwargs):
    
    trainset = BinaryBlobs(split='train',   image_transform=train_image_transform, 
        label_transform=train_label_transform, joint_transform=train_joint_transform, **kwargs)
    testset  = BinaryBlobs(split='test',    image_transform=test_image_transform,
        label_transform=test_label_transform, joint_transform=test_joint_transform, **kwargs)
    validset = BinaryBlobs(split='validate',image_transform=validate_image_transform, 
        label_transform=validate_label_transform, joint_transform=validate_joint_transform, **kwargs)


    trainloader = data.DataLoader(trainset, batch_size=train_batch_size,
                                            num_workers=num_workers)

    testloader = data.DataLoader(testset, batch_size=test_batch_size,
                                            num_workers=num_workers)

    validloader = data.DataLoader(validset, batch_size=test_batch_size,
                                            num_workers=num_workers)

    return trainloader, testloader, validloader

if __name__ == "__main__":
    ds = BinaryBlobs()
    ds[0]