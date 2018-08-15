"""
Regularized MNIST Example
================================

This example demonstrates adding and logging arbitrary regularization losses, in this case,
L2 activity regularization and L1 weight regularization.

- Add a `_losses` dictionary to any module containing loss names and values
- Use a criterion from `inferno.extensions.criteria.regularized` that will collect and add those losses
- Call `Trainer.observe_training_and_validation_states` to log the losses as well
"""

import argparse
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from inferno.extensions.layers.reshape import Flatten
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger


class RegularizedLinear(nn.Linear):
    def __init__(self, *args, ar_weight=1e-3, l1_weight=1e-3, **kwargs):
        super(RegularizedLinear, self).__init__(*args, **kwargs)
        self.ar_weight = ar_weight
        self.l1_weight = l1_weight
        self._losses = {}

    def forward(self, input):
        output = super(RegularizedLinear, self).forward(input)
        self._losses['activity_regularization'] = (output * output).sum() * self.ar_weight
        self._losses['l1_weight_regularization'] = torch.abs(self.weight).sum() * self.l1_weight
        return output


def model_fn():
    return nn.Sequential(
        Flatten(),
        RegularizedLinear(in_features=784, out_features=256),
        nn.LeakyReLU(),
        RegularizedLinear(in_features=256, out_features=128),
        nn.LeakyReLU(),
        RegularizedLinear(in_features=128, out_features=10)
    )


def mnist_data_loaders(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


def train_model(args):
    model = model_fn()
    train_loader, validate_loader = mnist_data_loaders(args)

    # Build trainer
    trainer = Trainer(model) \
        .build_criterion('RegularizedCrossEntropyLoss') \
        .build_metric('CategoricalError') \
        .build_optimizer('Adam') \
        .validate_every((1, 'epochs')) \
        .save_every((1, 'epochs')) \
        .save_to_directory(args.save_directory) \
        .set_max_num_epochs(args.epochs) \
        .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                        log_images_every='never'),
                      log_directory=args.save_directory)

    # Record regularization losses
    trainer.logger.observe_training_and_validation_states([
        'main_loss',
        'total_regularization_loss',
        'activity_regularization',
        'l1_weight_regularization'
    ])

    # Bind loaders
    trainer \
        .bind_loader('train', train_loader) \
        .bind_loader('validate', validate_loader)

    if args.cuda:
        trainer.cuda()

    # Go!
    trainer.fit()


def main(argv):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--save-directory', type=str, default='output/mnist/v1',
                        help='output directory')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args(argv)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    train_model(args)


if __name__ == '__main__':
    main(sys.argv[1:])
