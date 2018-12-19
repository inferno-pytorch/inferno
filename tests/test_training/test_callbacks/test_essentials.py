import unittest
import shutil
import h5py as h5
from os.path import dirname, join
from os import listdir
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.essentials import DumpHDF5Every
from inferno.utils.test_utils import generate_random_dataloader
from inferno.extensions.layers import Conv2D, AsMatrix
from torch.nn import Sequential, MaxPool2d, AdaptiveAvgPool2d, Linear, Softmax


class TestEssentials(unittest.TestCase):
    WORKING_DIRECTORY = dirname(__file__)

    def setUp(self):
        # Build a simple ass model
        model = Sequential(Conv2D(3, 8, 3, activation='ReLU'),
                           MaxPool2d(2, 2),
                           Conv2D(8, 8, 3, activation='ReLU'),
                           MaxPool2d(2, 2),
                           Conv2D(8, 8, 3, activation='ReLU'),
                           MaxPool2d(2, 2),
                           Conv2D(8, 8, 3, activation='ReLU'),
                           AdaptiveAvgPool2d((1, 1)),
                           AsMatrix(),
                           Linear(8, 10))

        train_dataloader = generate_random_dataloader(512, (3, 32, 32), 10, batch_size=16,
                                                      dtype='float32')
        validate_dataloader = generate_random_dataloader(32, (3, 32, 32), 10, batch_size=16,
                                                         dtype='float32')
        # Build trainer
        trainer = Trainer(model)\
            .bind_loader('train', train_dataloader)\
            .bind_loader('validate', validate_dataloader)\
            .save_to_directory(to_directory=join(self.WORKING_DIRECTORY, 'Weights'))\
            .build_criterion('CrossEntropyLoss').build_optimizer('RMSprop')
        self.trainer = trainer

    def test_dump_hdf5_every(self):
        # Configure callback
        dumper = DumpHDF5Every((1, 'epoch'),
                               to_directory=join(self.WORKING_DIRECTORY, 'Weights'),
                               dump_after_every_validation_run=True)
        self.trainer\
            .set_max_num_epochs(4)\
            .register_callback(dumper)\
            .validate_every((16, 'iterations'))

        self.trainer.fit()
        all_files = listdir(join(self.WORKING_DIRECTORY, 'Weights'))
        for epoch in range(5):
            self.assertIn('dump.training.epoch{}.iteration{}.h5'.format(epoch, epoch * 32),
                          all_files)
            # We don't validate at last epoch
            if epoch != 4:
                self.assertIn('dump.validation.epoch{}.iteration{}.h5'
                              .format(epoch, (epoch * 32) + 16),
                              all_files)
                self.assertIn('dump.validation.epoch{}.iteration{}.h5'
                              .format(epoch, (epoch * 32) + 32),
                              all_files)

        # Check if the keys are right in a training dump
        sample_file_path = join(self.WORKING_DIRECTORY, 'Weights',
                                'dump.training.epoch0.iteration0.h5')
        with h5.File(sample_file_path, 'r') as sample_file:
            all_dataset_names = list(sample_file.keys())
        self.assertSequenceEqual(all_dataset_names,
                                 ['training_inputs_0', 'training_prediction', 'training_target'])
        # Check if the keys are right in a validation dump
        sample_file_path = join(self.WORKING_DIRECTORY, 'Weights',
                                'dump.validation.epoch0.iteration16.h5')
        with h5.File(sample_file_path, 'r') as sample_file:
            all_dataset_names = list(sample_file.keys())
        self.assertSequenceEqual(all_dataset_names,
                                 ['validation_inputs_0', 'validation_prediction',
                                  'validation_target'])

    def tearDown(self):
        shutil.rmtree(join(self.WORKING_DIRECTORY, 'Weights'))


if __name__ == '__main__':
    unittest.main()
