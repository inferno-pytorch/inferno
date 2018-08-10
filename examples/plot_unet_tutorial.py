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
import pylab
import torch
import numpy

##############################################################################
# should CUDA be used
USE_CUDA = True


##############################################################################
# Dataset
# --------------
# For simplicity we will use a toy dataset where we need to perform
# a binary segmentation task.
from inferno.io.box.binary_blobs import get_binary_blob_loaders

# convert labels from long to float as needed by
# binary cross entropy loss
label_transform = lambda x : torch.from_numpy(x).float()

train_loader, test_loader, validate_loader = get_binary_blob_loaders(
    size=8, # how many images per {train,test,validate}
    train_batch_size=2,
    length=256, # <= size of the images
    gaussian_noise_sigma=1.5, # <= how noise are the images
    train_label_transform = label_transform,
    validate_label_transform = label_transform
)

image_channels = 1   # <-- number of channels of the image
pred_channels = 1  # <-- number of channels needed for the prediction

##############################################################################
# Visualize Dataset
# ~~~~~~~~~~~~~~~~~~~~~~
fig = pylab.figure()

for i,(image, target) in enumerate(train_loader):
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image[0,0,...])
    ax.set_title('raw data')
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(target[0,...])
    ax.set_title('ground truth')
    break
fig.tight_layout()
pylab.show()


##############################################################################
# Simple UNet
# ----------------------------
# We start with a very simple predefined
# res block UNet. By default, this UNet uses  ReLUs (in conjunction with batchnorm) as nonlinearities  
# With :code:`activated=False` we make sure that the last layer
# is not activated since we chain the UNet with a sigmoid
# activation function.
from inferno.extensions.layers.unet import ResBlockUNet
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
    ResBlockUNet(dim=2, in_channels=5, out_channels=pred_channels,  activated=False) ,
    RemoveSingletonDimension(dim=1),
    torch.nn.Sigmoid()
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
    trainer.build_criterion('BCELoss')
    trainer.build_optimizer('Adam')
    trainer.validate_every((kwargs.get('validate_every', 10), 'epochs'))
    trainer.save_every((kwargs.get('save_every', 10), 'epochs'))
    trainer.save_to_directory(ensure_dir(kwargs.get('save_dir', 'save_dor')))
    trainer.set_max_num_epochs(kwargs.get('max_num_epochs', 20))

    # bind the loaders
    trainer.bind_loader('train', loaders[0]) 
    trainer.bind_loader('validate', loaders[1])

    if USE_CUDA:
        trainer.cuda()

    # do the training
    trainer.fit()

    return trainer


trainer = train_model(model=model_a, loaders=[train_loader, validate_loader], save_dir='model_a')



##############################################################################
# Prediction
# ----------------------------
# The trainer contains the trained model and we can do predictions.
# We use :code:`unwrap` to convert the results to numpy arrays.
trainer.eval_mode()
from inferno.utils.torch_utils import unwrap


for image, target in test_loader:

    # transfer image to gpu
    image = image.cuda() if USE_CUDA else image

    # get batch size from image
    batch_size = image.size()[0]
    
    prediction = trainer.apply_model(image)

    image = unwrap(image,      as_numpy=True, to_cpu=True)
    prediction = unwrap(prediction, as_numpy=True, to_cpu=True)


    fig = pylab.figure()

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(image[0,0,...])
    ax.set_title('raw data')

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(target[0,...])
    ax.set_title('raw data')

    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(prediction[0,...])
    ax.set_title('raw data')

    fig.tight_layout()
    pylab.show()
