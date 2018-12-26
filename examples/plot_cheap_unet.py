"""
UNet Tutorial
================================
A unet example which can be run without a gpu
"""

##############################################################################
# Preface
# --------------
# We start with some unspectacular multi purpose imports needed for this example
import matplotlib.pyplot as plt
import torch
from torch import nn
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

if False:
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
    trainer.set_max_num_epochs(kwargs.get('max_num_epochs', 20))

    # bind the loaders
    trainer.bind_loader('train', loaders[0])
    trainer.bind_loader('validate', loaders[1])

    if USE_CUDA:
        trainer.cuda()

    # do the training
    trainer.fit()

    return trainer




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



##############################################################################
# Custom UNet
# ----------------------------
# Often one needs to have a UNet with custom layers.
# Here we show how to implement such a customized UNet.
# To this end we derive from :code:`UNetBase`.
# For the sake of this example we will create
# a Unet which uses depthwise convolutions and might be trained on a CPU
from inferno.extensions.models import UNetBase
from inferno.extensions.layers import ConvSELU2D, ConvReLU2D, ConvELU2D, ConvSigmoid2D,Conv2D,ConvActivation


class CheapConv(nn.Module):
    def __init__(self, in_channels, out_channels, activated):
        super(CheapConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activated:
            self.convs = torch.nn.Sequential(
                ConvActivation(in_channels=in_channels, out_channels=in_channels, depthwise=True, kernel_size=(3, 3), activation='ReLU', dim=2),
                ConvReLU2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1))
            )
        else:
            self.convs = torch.nn.Sequential(
                ConvActivation(in_channels=in_channels, out_channels=in_channels, depthwise=True, kernel_size=(3, 3), activation='ReLU', dim=2),
                Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1))
            )
    def forward(self, x):
        assert x.shape[1] == self.in_channels,"input has wrong number of channels"
        x =  self.convs(x)
        assert x.shape[1] == self.out_channels,"output has wrong number of channels"
        return x 


class CheapConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activated):
        super(CheapConvBlock, self).__init__()
        self.activated = activated
        self.in_channels = in_channels
        self.out_channels = out_channels
        if(in_channels != out_channels):
            self.start = ConvReLU2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1))
        else:
            self.start = None
        self.conv_a = CheapConv(in_channels=out_channels, out_channels=out_channels, activated=True)
        self.conv_b = CheapConv(in_channels=out_channels, out_channels=out_channels, activated=False)
        self.activation = torch.nn.ReLU()
    def forward(self, x):
        x_input = x
        if self.start is not None:
            x_input = self.start(x_input)

        x = self.conv_a(x_input)
        x = self.conv_b(x)

        x = x + x_input

        if self.activated:
            x = self.activation(x)
        return x

class MySimple2DCpUnet(UNetBase):
    def __init__(self, in_channels, out_channels, depth=3, residual=False, **kwargs):
        super(MySimple2DCpUnet, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             dim=2, depth=depth, **kwargs)

    def conv_op_factory(self, in_channels, out_channels, part, index):

        # last? 
        last = part == 'up' and index==0
        return CheapConvBlock(in_channels=in_channels, out_channels=out_channels, activated=not last),False



from inferno.extensions.layers import RemoveSingletonDimension
model_b = torch.nn.Sequential(
    CheapConv(in_channels=image_channels, out_channels=4, activated=True),
    MySimple2DCpUnet(in_channels=4, out_channels=pred_channels) ,
    RemoveSingletonDimension(dim=1)
)


###################################################
# do the training (with the same functions as before)
trainer = train_model(model=model_b, loaders=[train_loader, validate_loader], save_dir='model_b', lr=0.001)

###################################################
# do the training (with the same functions as before)1
predict(trainer=trainer, test_loader=test_loader)

