"""
UNet Tutorial
================================
A tentative tutorial on the usage
of the unet framework in inferno
"""

##############################################################################
# Preface
# --------------
# We start with some unspectacular multi purpose imports needed for this example
import matplotlib.pyplot as plt
import torch
import numpy

##############################################################################

# determine whether we have a gpu
# and should use cuda
USE_CUDA = torch.cuda.is_available()


##############################################################################
# Dataset
# --------------
# For simplicity we will use a toy dataset where we need to perform
# a binary segmentation task.
from inferno.io.box.binary_blobs import get_binary_blob_loaders

# convert labels from long to float as needed by
# binary cross entropy loss
def label_transform(x):
    return torch.from_numpy(x).float()
#label_transform = lambda x : torch.from_numpy(x).float()

train_loader, test_loader, validate_loader = get_binary_blob_loaders(
    size=8, # how many images per {train,test,validate}
    train_batch_size=2,
    length=256, # <= size of the images
    gaussian_noise_sigma=1.4, # <= how noise are the images
    train_label_transform = label_transform,
    validate_label_transform = label_transform
)

image_channels = 1   # <-- number of channels of the image
pred_channels = 1  # <-- number of channels needed for the prediction

##############################################################################
# Visualize Dataset
# ~~~~~~~~~~~~~~~~~~~~~~
fig = plt.figure()

for i,(image, target) in enumerate(train_loader):
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image[0,0,...])
    ax.set_title('raw data')
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(target[0,...])
    ax.set_title('ground truth')
    break
fig.tight_layout()
plt.show()


##############################################################################
# Simple UNet
# ----------------------------
# We start with a very simple predefined
# res block UNet. By default, this UNet uses  ReLUs (in conjunction with batchnorm) as nonlinearities
# With :code:`activated=False` we make sure that the last layer
# is not activated since we chain the UNet with a sigmoid
# activation function.
from inferno.extensions.models import ResBlockUNet
from inferno.extensions.layers import RemoveSingletonDimension

model = torch.nn.Sequential(
    ResBlockUNet(dim=2, in_channels=image_channels, out_channels=pred_channels,  activated=False),
    RemoveSingletonDimension(dim=1),
    torch.nn.Sigmoid()
)

##############################################################################
# while the model above will work in principal, it has some drawbacks.
# Within the UNet, the number of features is increased by a multiplicative
# factor while going down, the so-called gain. The default value for the gain is 2.
# Since we start with only a single channel we could either increase the gain,
# or use a some convolutions to increase the number of channels
# before the the UNet.
from inferno.extensions.layers import ConvReLU2D
model_a = torch.nn.Sequential(
    ConvReLU2D(in_channels=image_channels, out_channels=5, kernel_size=3),
    ResBlockUNet(dim=2, in_channels=5, out_channels=pred_channels,  activated=False,
        res_block_kwargs=dict(batchnorm=True,size=2)) ,
    RemoveSingletonDimension(dim=1)
    # torch.nn.Sigmoid()
)





##############################################################################
# Training
# ----------------------------
# To train the unet, we use the infernos Trainer class of inferno.
# Since we train many models later on in this example we encapsulate
# the training in a function (see :ref:`sphx_glr_auto_examples_trainer.py` for
# an example dedicated to the trainer itself).
from inferno.trainers import Trainer
from inferno.utils.python_utils import ensure_dir

def train_model(model, loaders, **kwargs):

    trainer = Trainer(model)
    trainer.build_criterion('BCEWithLogitsLoss')
    trainer.build_optimizer('Adam', lr=kwargs.get('lr', 0.0001))
    #trainer.validate_every((kwargs.get('validate_every', 10), 'epochs'))
    #trainer.save_every((kwargs.get('save_every', 10), 'epochs'))
    #trainer.save_to_directory(ensure_dir(kwargs.get('save_dir', 'save_dor')))
    trainer.set_max_num_epochs(kwargs.get('max_num_epochs', 200))

    # bind the loaders
    trainer.bind_loader('train', loaders[0])
    trainer.bind_loader('validate', loaders[1])

    if USE_CUDA:
        trainer.cuda()

    # do the training
    trainer.fit()

    return trainer


trainer = train_model(model=model_a, loaders=[train_loader, validate_loader], save_dir='model_a', lr=0.01)



##############################################################################
# Prediction
# ----------------------------
# The trainer contains the trained model and we can do predictions.
# We use :code:`unwrap` to convert the results to numpy arrays.
# Since we want to do many prediction we encapsulate the
# the prediction in a function
from inferno.utils.torch_utils import unwrap

def predict(trainer, test_loader,  save_dir=None):


    trainer.eval_mode()
    for image, target in test_loader:

        # transfer image to gpu
        image = image.cuda() if USE_CUDA else image

        # get batch size from image
        batch_size = image.size()[0]

        for b in range(batch_size):
            prediction = trainer.apply_model(image)
            prediction = torch.nn.functional.sigmoid(prediction)

            image = unwrap(image,      as_numpy=True, to_cpu=True)
            prediction = unwrap(prediction, as_numpy=True, to_cpu=True)
            target = unwrap(target, as_numpy=True, to_cpu=True)

            fig = plt.figure()

            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(image[b,0,...])
            ax.set_title('raw data')

            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(target[b,...])
            ax.set_title('ground truth')

            ax = fig.add_subplot(2, 2, 4)
            ax.imshow(prediction[b,...])
            ax.set_title('prediction')

            fig.tight_layout()
            plt.show()

###################################################
# do the prediction
predict(trainer=trainer, test_loader=test_loader)




##############################################################################
# Custom UNet
# ----------------------------
# Often one needs to have a UNet with custom layers.
# Here we show how to implement such a customized UNet.
# To this end we derive from :code:`UNetBase`.
# For the sake of this example we will create
# a rather exotic UNet which uses different types
# of convolutions/non-linearities in the different branches
# of the unet
from inferno.extensions.models import UNetBase
from inferno.extensions.layers import ConvSELU2D, ConvReLU2D, ConvELU2D, ConvSigmoid2D,Conv2D
from inferno.extensions.layers.sampling import Upsample

class MySimple2DUnet(UNetBase):
    def __init__(self, in_channels, out_channels, depth=3, **kwargs):
        super(MySimple2DUnet, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             dim=2, depth=depth, **kwargs)

    def conv_op_factory(self, in_channels, out_channels, part, index):

        if part == 'down':
            return torch.nn.Sequential(
                ConvELU2D(in_channels=in_channels,  out_channels=out_channels, kernel_size=3),
                ConvELU2D(in_channels=out_channels,  out_channels=out_channels, kernel_size=3)
            ), False
        elif part == 'bottom':
            return torch.nn.Sequential(
                ConvReLU2D(in_channels=in_channels,  out_channels=out_channels, kernel_size=3),
                ConvReLU2D(in_channels=out_channels,  out_channels=out_channels, kernel_size=3),
            ), False
        elif part == 'up':
            # are we in the very last block?
            if index  == 0:
                return torch.nn.Sequential(
                    ConvELU2D(in_channels=in_channels,  out_channels=out_channels, kernel_size=3),
                    Conv2D(in_channels=out_channels,  out_channels=out_channels, kernel_size=3)
                ), False
            else:
                return torch.nn.Sequential(
                    ConvELU2D(in_channels=in_channels,   out_channels=out_channels, kernel_size=3),
                    ConvReLU2D(in_channels=out_channels,  out_channels=out_channels, kernel_size=3)
                ), False
        else:
            raise RuntimeError("something is wrong")




    # this function CAN be implemented, if not, MaxPooling is used by default
    def downsample_op_factory(self, index):
        return torch.nn.MaxPool2d(kernel_size=2, stride=2)

    # this function CAN be implemented, if not, Upsampling is used by default
    def upsample_op_factory(self, index):
        return Upsample(mode='bilinear', align_corners=False,scale_factor=2)

model_b = torch.nn.Sequential(
    ConvReLU2D(in_channels=image_channels, out_channels=5, kernel_size=3),
    MySimple2DUnet(in_channels=5, out_channels=pred_channels) ,
    RemoveSingletonDimension(dim=1)
)


###################################################
# do the training (with the same functions as before)
trainer = train_model(model=model_b, loaders=[train_loader, validate_loader], save_dir='model_b', lr=0.001)

###################################################
# do the training (with the same functions as before)
predict(trainer=trainer, test_loader=test_loader)

