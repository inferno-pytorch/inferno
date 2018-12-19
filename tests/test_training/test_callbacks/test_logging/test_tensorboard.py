import unittest

import os
import numpy as np
import torch
import torch.nn as nn
from inferno.trainers.basic import Trainer
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.extensions.layers.reshape import AsMatrix


class TestTensorboard(unittest.TestCase):
    ROOT_DIR = os.path.dirname(__file__)
    PRECISION = 'float'

    @staticmethod
    def _make_test_model(input_channels):
        toy_net = nn.Sequential(nn.Conv2d(input_channels, 128, 3, 1, 1),
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

    def setUp(self, input_channels=3):
        # Build model
        net = self._make_test_model(input_channels)

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
            .set_precision(self.PRECISION)

        train_loader, test_loader = self.getRandomDataloaders(input_channels=input_channels)

        # Bind loaders
        self.trainer.bind_loader('train', train_loader).bind_loader('validate', test_loader)

    def getRandomDataloaders(self, input_channels=3):
        # Convert build random tensor dataset
        data_shape = (1, input_channels, 64, 64)
        target_shape = (1)
        random_array = torch.from_numpy(np.random.rand(*data_shape)).float()
        target_array = torch.from_numpy(np.random.randint(0, 9, size=target_shape))
        train_dataset = TensorDataset(random_array, target_array)
        test_dataset = TensorDataset(random_array, target_array)

        # Build dataloaders from dataset
        train_loader = DataLoader(train_dataset, batch_size=1,
                                  shuffle=True, num_workers=1, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=True, num_workers=1, pin_memory=False)
        return train_loader, test_loader

    def test_tensorboard(self):
        # Set up if required
        if not hasattr(self, 'trainer'):
            self.setUp(input_channels=3)
        # Train
        self.trainer.fit()
        # Print info for check
        self.trainer.print("Inspect logs at: {}".format(self.trainer.log_directory))

    def test_tensorboard_grayscale(self):
        # Set up if required
        if not hasattr(self, 'trainer'):
            self.setUp(input_channels=1)
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
        train_loader, test_loader = self.getRandomDataloaders(input_channels=3)
        trainer.bind_loader('train', train_loader).bind_loader('validate', test_loader)
        trainer.fit()
        trainer.print("Inspect logs at: {}".format(self.trainer.log_directory))


if __name__ == '__main__':
    unittest.main()
