"""
Train UNet Example
================================

This example should illustrate how to use the trainer class
in conjunction with a unet, we use a toy dataset here

"""


import torch.nn as nn
from inferno.io.box.binary_blobs import get_binary_blob_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from inferno.extensions.layers.building_blocks import ResBlock
from inferno.extensions.layers.unet import ResBlockUNet
from inferno.utils.torch_utils import unwrap

import pylab

# lil helper to make sure dirs exits
import os
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# change directories to your needs
LOG_DIRECTORY = ensure_dir('log')
SAVE_DIRECTORY = ensure_dir('save')
DATASET_DIRECTORY = ensure_dir('dataset')


USE_CUDA = True

# Build a residual unet where the last layer is not activated
model = nn.Sequential(
    ResBlock(dim=2, in_channels=1, out_channels=5),
    ResBlockUNet(dim=2, in_channels=5, out_channels=2,  activated=False) 
)
train_loader, test_loader, validate_loader = get_binary_blob_loaders(
    train_batch_size=3,
    length=512, # <= size of the images


# Build trainer
trainer = Trainer(model) \
.build_criterion('CrossEntropyLoss') \
.build_metric('IOU') \
.build_optimizer('Adam') \
.validate_every((10, 'epochs')) \
.save_every((10, 'epochs')) \
.save_to_directory(SAVE_DIRECTORY) \
.set_max_num_epochs(40) \

# Bind loaders
trainer \
    .bind_loader('train', train_loader) \
    .bind_loader('validate', validate_loader)

if USE_CUDA:
    trainer.cuda()

# Go!
trainer.fit()


# predict:
trainer.load(best=True)
trainer\
    .bind_loader('train', train_loader) \
    .bind_loader('validate', validate_loader)
trainer.eval_mode()


trainer.cuda()

# look at an example
for x,y in test_loader:
    if USE_CUDA:
        x = x.cuda()
    yy = trainer.apply_model(x)
    yy = nn.functional.softmax(yy,dim=1)
    yy = unwrap(yy, as_numpy=True, to_cpu=True)
    x  = unwrap(x,  as_numpy=True, to_cpu=True)
    y  = unwrap(y, as_numpy=True, to_cpu=True)

    batch_size = yy.shape[0]
    for b in range(batch_size):

        fig = pylab.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(x[b,0,...])
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(y[b,...])
        ax3 = fig.add_subplot(1,3,3)
        ax3.imshow(yy[b,1,...])

        pylab.show()

    break