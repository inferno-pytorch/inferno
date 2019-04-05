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
# For simplicity we will use a toy dataset based on a noisy sine
from inferno.io.box.noisy_func_1d import get_noisy_func_loader

import math
def f(i):
    data = math.sin(float(i)/10.0)
    return data, int(data>0.0)

# convert labels from long to float as needed by
# binary cross entropy loss
def label_transform(x):
    return torch.from_numpy(x).float()

train_loader, test_loader, validate_loader = get_noisy_func_loader(
    f=f,
    size=20, # how many items per {train,test,validate}
    train_batch_size=20,
    test_batch_size=20,
    signal_length=256, # <= length of array
    sigma=1.2,
    label_transform=label_transform
)

image_channels = 1   # <-- number of channels of the image
pred_channels = 1  # <-- number of channels needed for the prediction

if False:

    # ##############################################################################
    # # Visualize Dataset
    # # ~~~~~~~~~~~~~~~~~~~~~~
    fig = plt.figure()
    for i,(noisy_signal, gt) in enumerate(train_loader):
        ax = fig.add_subplot(2, 2, i+1)
        ax.plot(numpy.arange(noisy_signal.size()[2]), noisy_signal.numpy()[0,0,...])#, 'ro')
        ax.plot(numpy.arange(gt.size()[2]), gt.numpy()[0,0,...], 'r')
        if i>=3 :
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
    trainer.validate_every((kwargs.get('validate_every', 1), 'epochs'))
    #trainer.save_every((kwargs.get('save_every', 10), 'epochs'))
    #trainer.save_to_directory(ensure_dir(kwargs.get('save_dir', 'save_dor')))
    trainer.set_max_num_epochs(kwargs.get('max_num_epochs', 4))

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
    c = 0
    for noisy_signal, gt in test_loader:
        if c > 2:
            break
        # transfer noisy_signal to gpu
        noisy_signal = noisy_signal.cuda() if USE_CUDA else noisy_signal

        # get batch size from noisy_signal
        batch_size = noisy_signal.size()[0]

        prediction = trainer.apply_model(noisy_signal)
        prediction = torch.nn.functional.sigmoid(prediction)

        noisy_signal = unwrap(noisy_signal,      as_numpy=True, to_cpu=True)
        prediction = unwrap(prediction, as_numpy=True, to_cpu=True)
        gt = unwrap(gt, as_numpy=True, to_cpu=True)

        for b in range(batch_size):
            fig = plt.figure()
            rng = numpy.arange(prediction[b,...].size)

            ax = fig.add_subplot(1, 1, 1)
            ax.plot(rng,noisy_signal[b,0,...],'r' , rng, gt[b,0,...],'g', rng, prediction[b,0,...],'b')
            fig.tight_layout()
            plt.show()

            c += 1
            if c > 2:
                break


##############################################################################
# Custom UNet
# ----------------------------
# Often one needs to have a UNet with custom layers.
# Here we show how to implement such a customized UNet.
# To this end we derive from :code:`UNetBase`.
# For the sake of this example we will create
# a Unet which uses depthwise convolutions and might be trained on a CPU
from inferno.extensions.models import *
from inferno.extensions.layers import *


unet = UNet(in_channels=1, initial_features=64, out_channels=1, dim=1, depth=2)

##################################################
# do the training 
trainer = train_model(model=unet, loaders=[train_loader, validate_loader], save_dir='unet', lr=0.001)

###################################################
# visualizer predictions
predict(trainer=trainer, test_loader=test_loader)

