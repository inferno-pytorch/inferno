"""
Trainer Example
================================

This example should illustrate how to use the trainer class.

"""

import torch.nn as nn
from inferno.io.box.cifar import get_cifar10_loaders
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers import ConvELU2D
from inferno.extensions.layers import Flatten
from inferno.utils.python_utils import ensure_dir

from inferno.extensions.layers import SELU

##################################################
# change directories to your needs
LOG_DIRECTORY = ensure_dir('log')
SAVE_DIRECTORY = ensure_dir('save')
DATASET_DIRECTORY = ensure_dir('dataset')

##################################################
# shall models be downloaded
DOWNLOAD_CIFAR = True
USE_CUDA = True

##################################################
# Build torch model
model = nn.Sequential(
    ConvELU2D(in_channels=3, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    nn.Linear(in_features=(256 * 4 * 4), out_features=10),
    nn.Softmax()
)

##################################################
# data loaders
train_loader, validate_loader = get_cifar10_loaders(DATASET_DIRECTORY,
                                        download=DOWNLOAD_CIFAR)

##################################################
# Build trainer
trainer = Trainer(model)
trainer.build_criterion('CrossEntropyLoss')
trainer.build_metric('CategoricalError')
trainer.build_optimizer('Adam')
trainer.validate_every((2, 'epochs'))
trainer.save_every((5, 'epochs'))
trainer.save_to_directory(SAVE_DIRECTORY)
trainer.set_max_num_epochs(10)
trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                log_images_every='never'), 
              log_directory=LOG_DIRECTORY)

##################################################
# Bind loaders
trainer.bind_loader('train', train_loader)
trainer.bind_loader('validate', validate_loader)

##################################################
# activate cuda
if USE_CUDA:
    trainer.cuda()

##################################################
# fit
trainer.fit()
