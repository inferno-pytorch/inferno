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
# Dataset
# --------------
# For simplicity we will use a toy dataset where we need to perform
# a binary segmentation task.
from inferno.io.box.binary_blobs import get_binary_blob_loaders

# lambda to convert labels from long to float
# as need by binary cross entropy  loss
label_transform = lambda x : x.float()

train_loader, test_loader, validate_loader = get_binary_blob_loaders(
    train_batch_size=3,
    length=512, # <= size of the images
    gaussian_noise_sigma=1.5, # <= how noise are the images
    train_label_transform = label_transform,
    validate_label_transform = label_transform
)

##############################################################################
# Dataset
# --------------
# For simplicity we will use a toy dataset where we need to perform
# a binary segmentation task.