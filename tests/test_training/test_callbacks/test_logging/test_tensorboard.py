import unittest

import os
from shutil import rmtree
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
    SAVE_DIRECTORY = os.path.join(ROOT_DIR, 'saves')
    LOG_DIRECTORY = os.path.join(ROOT_DIR, 'logs')

    @staticmethod
    def _make_test_model(input_channels):
        toy_net = nn.Sequential(nn.Conv2d(input_channels, 8, 3, 1, 1),
                                nn.ELU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(8, 8, 3, 1, 1),
                                nn.ELU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(8, 16, 3, 1, 1),
                                nn.ELU(),
                                nn.AdaptiveMaxPool2d((1, 1)),
                                AsMatrix(),
                                nn.Linear(16, 10))
        return toy_net

    def tearDown(self):
        for d in [self.SAVE_DIRECTORY, self.LOG_DIRECTORY]:
            try:
                rmtree(d)
            except OSError:
                pass

    def get_random_dataloaders(self, input_channels=3):
        # Convert build random tensor dataset
        data_shape = (1, input_channels, 64, 64)
        target_shape = (1)
        random_array = torch.from_numpy(np.random.rand(*data_shape)).float()
        target_array = torch.from_numpy(np.random.randint(0, 9, size=target_shape))
        train_dataset = TensorDataset(random_array, target_array)
        test_dataset = TensorDataset(random_array, target_array)

        # Build dataloaders from dataset
        train_loader = DataLoader(train_dataset, batch_size=1,
                                  shuffle=True, num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=1,
                                 shuffle=True, num_workers=0, pin_memory=False)
        return train_loader, test_loader

    def get_trainer(self, input_channels):
        # Build model
        net = self._make_test_model(input_channels)
        # Build trainer
        trainer = Trainer(net)\
            .build_logger(TensorboardLogger(send_image_at_batch_indices=0,
                                            send_image_at_channel_indices='all',
                                            log_images_every=(20, 'iterations')),
                          log_directory=self.LOG_DIRECTORY)\
            .build_criterion('CrossEntropyLoss')\
            .build_metric('CategoricalError')\
            .build_optimizer('Adam')\
            .validate_every((1, 'epochs'))\
            .save_every((2, 'epochs'), to_directory=self.SAVE_DIRECTORY)\
            .save_at_best_validation_score()\
            .set_max_num_epochs(2)\
            .set_precision(self.PRECISION)
        # Bind loaders
        train_loader, test_loader = self.get_random_dataloaders(input_channels=input_channels)
        trainer.bind_loader('train', train_loader).bind_loader('validate', test_loader)
        return trainer

    def test_tensorboard(self):
        trainer = self.get_trainer(3)
        trainer.fit()

    def test_tensorboard_grayscale(self):
        trainer = self.get_trainer(1)
        trainer.fit()

    def test_serialization(self):
        trainer = self.get_trainer(3)
        # Serialize
        trainer.save()
        # Unserialize
        trainer = Trainer().load(os.path.join(self.ROOT_DIR, 'saves'))
        train_loader, test_loader = self.get_random_dataloaders(input_channels=3)
        trainer.bind_loader('train', train_loader).bind_loader('validate', test_loader)
        trainer.fit()


if __name__ == '__main__':
    unittest.main()
