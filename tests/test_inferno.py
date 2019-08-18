#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `inferno` package."""


import unittest
import numpy as np
import torch
import os
import shutil
from os.path import dirname, join
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader
from inferno.extensions.layers import Conv2D, BNReLUConv2D
from inferno.extensions.layers import AsMatrix
from inferno.extensions.containers import Graph
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.essentials import NaNDetector
from inferno.trainers.callbacks.base import Callback
from torch import nn


class TestInferno(unittest.TestCase):
    """Tests for `inferno` package."""

    NUM_SAMPLES = 100
    NUM_TRAINING_SAMPLES = 70
    NUM_CLASSES = 10
    WORKING_DIRECTORY = dirname(__file__)

    def read_environment_variables(self):
        self.NUM_SAMPLES = int(os.getenv('INFERNO_TEST_NUM_SAMPLES', str(self.NUM_SAMPLES)))
        self.NUM_TRAINING_SAMPLES = int(os.getenv('INFERNO_TEST_NUM_SAMPLES',
                                                  str(self.NUM_TRAINING_SAMPLES)))
        self.NUM_CLASSES = int(os.getenv('INFERNO_TEST_NUM_CLASSES', str(self.NUM_CLASSES)))
        self.WORKING_DIRECTORY = os.getenv('INFERNO_TEST_WORKING_DIRECTORY',
                                           self.WORKING_DIRECTORY)

    def setUp(self):
        """Set up test fixtures, if any."""
        self.setUpDatasets()

    def setUpDatasets(self):
        # Build training dataset
        inputs, targets = self.generate_random_data(self.NUM_SAMPLES, (3, 32, 32),
                                                    num_classes=self.NUM_CLASSES,
                                                    dtype='float32')
        # Split to train and split
        train_inputs, train_targets = inputs[:self.NUM_TRAINING_SAMPLES], \
                                      targets[:self.NUM_TRAINING_SAMPLES]
        validate_inputs, validate_targets = inputs[self.NUM_TRAINING_SAMPLES:], \
                                            targets[self.NUM_TRAINING_SAMPLES:]
        # Convert to tensor and build dataset
        train_dataset = TensorDataset(torch.from_numpy(train_inputs),
                                      torch.from_numpy(train_targets))
        validate_dataset = TensorDataset(torch.from_numpy(validate_inputs),
                                         torch.from_numpy(validate_targets))
        # Build dataloaders from dataset
        self.train_loader = DataLoader(train_dataset, batch_size=16,
                                       shuffle=True, num_workers=0, pin_memory=False)
        self.validate_loader = DataLoader(validate_dataset, batch_size=16,
                                          shuffle=True, num_workers=0, pin_memory=False)

    def setUpCallbacks(self):

        class RecordSaveInfo(Callback):
            def __init__(self):
                super(RecordSaveInfo, self).__init__()
                self.best_saves_at_iteration_epoch = []
                self.saves_at_iteration_epoch = []

            def begin_of_save(self, epoch_count, iteration_count,
                              is_iteration_with_best_validation_score, **_):
                if is_iteration_with_best_validation_score:
                    self.best_saves_at_iteration_epoch.append((iteration_count, epoch_count))
                else:
                    self.saves_at_iteration_epoch.append((iteration_count, epoch_count))

        self.RecordSaveInfo = RecordSaveInfo

    def generate_random_data(self, num_samples, shape, num_classes,
                             hardness=0.3, dtype=None):
        dataset_input = np.zeros((num_samples,) + shape, dtype=dtype)
        dataset_target = np.random.randint(num_classes, size=num_samples)
        for sample_num in range(num_samples):
            dataset_input[sample_num] = np.random.normal(loc=dataset_target[sample_num],
                                                         scale=(1 - hardness),
                                                         size=shape)
        return dataset_input, dataset_target

    def tearDown(self):
        """Tear down test fixtures, if any."""
        if os.path.exists(join(self.WORKING_DIRECTORY, 'Weights')):
            shutil.rmtree(join(self.WORKING_DIRECTORY, 'Weights'))

    def build_graph_model(self):
        model = Graph()
        model\
            .add_input_node('input')\
            .add_node('conv1', Conv2D(3, 8, 3), 'input')\
            .add_node('conv2', BNReLUConv2D(8, 8, 3), 'conv1')\
            .add_node('pool1', nn.MaxPool2d(kernel_size=2, stride=2), 'conv2')\
            .add_node('conv3', BNReLUConv2D(8, 8, 3), 'pool1')\
            .add_node('pool2', nn.MaxPool2d(kernel_size=2, stride=2), 'conv3')\
            .add_node('conv4', BNReLUConv2D(8, 8, 3), 'pool2')\
            .add_node('pool3', nn.AdaptiveAvgPool2d(output_size=(1, 1)), 'conv4')\
            .add_node('matrix', AsMatrix(), 'pool3')\
            .add_node('linear', nn.Linear(8, self.NUM_CLASSES), 'matrix')\
            .add_output_node('output', 'linear')
        return model

    def test_training_cpu(self):
        """Test Trainer."""
        # Build model
        model = self.build_graph_model()

        # Build callbacks
        # save_info_recorder = RecordSaveInfo()
        # Build trainer
        trainer = Trainer(model)\
            .save_every((2, 'epochs'), to_directory=join(self.WORKING_DIRECTORY, 'Weights'))\
            .validate_every((100, 'iterations'), for_num_iterations=10)\
            .set_max_num_epochs(4)\
            .save_at_best_validation_score()\
            .build_optimizer('RMSprop')\
            .build_criterion('CrossEntropyLoss')\
            .build_metric('CategoricalError')\
            .register_callback(NaNDetector)
        # Bind datasets
        trainer\
            .bind_loader('train', self.train_loader)\
            .bind_loader('validate', self.validate_loader)
        # Go
        trainer.pickle_module = 'dill'
        trainer.fit()



if __name__ == '__main__':
    unittest.main()
