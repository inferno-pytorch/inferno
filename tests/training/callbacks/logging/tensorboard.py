import unittest

import os
import torch.nn as nn
from inferno.trainers.basic import Trainer
from inferno.io.box.cifar10 import get_cifar10_loaders
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.reshape import AsMatrix


class TestTensorboard(unittest.TestCase):
    ROOT_DIR = os.path.dirname(__file__)
    PRECISION = 'half'
    DOWNLOAD_CIFAR = True

    @staticmethod
    def _make_test_model():
        toy_net = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                                nn.ELU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(128, 128, 3, 1, 1),
                                nn.ELU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(128, 256, 3, 1, 1),
                                nn.ELU(),
                                nn.AdaptiveMaxPool2d((1, 1)),
                                AsMatrix(),
                                nn.Linear(256, 10),
                                nn.Softmax())
        return toy_net

    def setUp(self):
        # Build model
        net = self._make_test_model()

        # Build trainer
        self.trainer = Trainer(net)\
            .build_logger(TensorboardLogger(send_image_at_batch_indices=0,
                                            send_image_at_channel_indices='all',
                                            log_images_every=(20, 'iterations')),
                          log_directory=os.path.join(self.ROOT_DIR, 'logs'))\
            .build_criterion('CrossEntropyLoss')\
            .build_metric('CategoricalError')\
            .build_optimizer('Adam')\
            .validate_every((1, 'epochs'))\
            .save_every((2, 'epochs'), to_directory=os.path.join(self.ROOT_DIR, 'saves'))\
            .save_at_best_validation_score()\
            .set_max_num_epochs(2)\
            .cuda().set_precision(self.PRECISION)

        # Load CIFAR10 data
        train_loader, test_loader = \
            get_cifar10_loaders(root_directory=os.path.join(self.ROOT_DIR, 'data'),
                                download=self.DOWNLOAD_CIFAR)

        # Bind loaders
        self.trainer.bind_loader('train', train_loader).bind_loader('validate', test_loader)

    def test_tensorboard(self):
        # Set up if required
        if not hasattr(self, 'trainer'):
            self.setUp()
        # Train
        self.trainer.fit()
        # Print info for check
        self.trainer.print("Inspect logs at: {}".format(self.trainer.log_directory))

    def test_serialization(self):
        if not hasattr(self, 'trainer'):
            self.setUp()
        # Serialize
        self.trainer.save()
        # Unserialize
        trainer = Trainer().load(os.path.join(self.ROOT_DIR, 'saves'))
        train_loader, test_loader = \
            get_cifar10_loaders(root_directory=os.path.join(self.ROOT_DIR, 'data'),
                                download=self.DOWNLOAD_CIFAR)
        trainer.bind_loader('train', train_loader).bind_loader('validate', test_loader)
        trainer.fit()
        trainer.print("Inspect logs at: {}".format(self.trainer.log_directory))


if __name__ == '__main__':
    TestTensorboard().test_tensorboard()
    # unittest.main()
