from unittest import TestCase, skipUnless
import torch
from unittest import main
import time
from os.path import join, dirname


class TestTrainer(TestCase):
    # Parameters
    ROOT_DIR = dirname(__file__)
    CUDA = False
    HALF_PRECISION = False
    DOWNLOAD_CIFAR = True

    @staticmethod
    def _make_test_model():
        import torch.nn as nn
        from inferno.extensions.layers.reshape import AsMatrix

        toy_net = nn.Sequential(nn.Conv2d(3, 8, 3, 1, 1),
                                nn.ELU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(8, 8, 3, 1, 1),
                                nn.ELU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(8, 16, 3, 1, 1),
                                nn.ELU(),
                                nn.AdaptiveAvgPool2d((1, 1)),
                                AsMatrix(),
                                nn.Linear(16, 10))
        return toy_net

    def test_cifar(self):
        from inferno.trainers.basic import Trainer
        from inferno.io.box.cifar import get_cifar10_loaders
        # Build cifar10 loaders
        trainloader, testloader = get_cifar10_loaders(root_directory=join(self.ROOT_DIR, 'data'),
                                                      download=self.DOWNLOAD_CIFAR)
        # Make model
        net = self._make_test_model()
        tic = time.time()
        # Make trainer
        trainer = Trainer(model=net)\
            .build_optimizer('Adam')\
            .build_criterion('CrossEntropyLoss')\
            .build_metric('CategoricalError')\
            .validate_every((1, 'epochs'))\
            .save_every((1, 'epochs'), to_directory=join(self.ROOT_DIR, 'saves'))\
            .save_at_best_validation_score()\
            .set_max_num_epochs(2)
        # Bind trainer to datasets
        trainer.bind_loader('train', trainloader).bind_loader('validate', testloader)
        # Check device and fit
        if self.CUDA:
            if self.HALF_PRECISION:
                trainer.cuda().set_precision('half').fit()
            else:
                trainer.cuda().fit()
        else:
            trainer.fit()
        toc = time.time()
        print("[*] Elapsed time: {} seconds.".format(toc - tic))

    def test_multi_io(self):
        from torch.utils.data.dataset import Dataset
        from torch.utils.data.dataloader import DataLoader
        from inferno.trainers.basic import Trainer

        class DummyDataset(Dataset):
            def __len__(self):
                return 42

            def __getitem__(self, item):
                # 2 inputs and 3 targets (say)
                return torch.rand(3, 32, 32), \
                       torch.rand(3, 32, 32), \
                       torch.rand(1).uniform_(), \
                       torch.rand(1).uniform_(), \
                       torch.rand(1).uniform_()

        class DummyNetwork(torch.nn.Module):
            def __init__(self):
                super(DummyNetwork, self).__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)

            def forward(self, *inputs):
                assert len(inputs) == 2
                out = self.conv(inputs[0])
                return out.view(inputs[0].size(0), -1).mean(1), \
                       out.view(inputs[0].size(0), -1).mean(1), \
                       out.view(inputs[0].size(0), -1).mean(1)

        class DummyCriterion(torch.nn.Module):
            def forward(self, predictions, targets):
                assert len(predictions) == len(targets) == 3
                return predictions[0].mean()

        loader = DataLoader(DummyDataset())
        net = DummyNetwork()

        trainer = Trainer(net)\
            .build_criterion(DummyCriterion)\
            .build_optimizer('Adam')\
            .set_max_num_iterations(50)\
            .bind_loader('train', loader, num_inputs=2, num_targets=3)

        trainer.fit()

    def test_serialization(self):
        from inferno.trainers.basic import Trainer
        import os

        # Make model
        net = self._make_test_model()
        # Make trainer
        trainer = Trainer(model=net) \
            .build_optimizer('Adam') \
            .build_criterion('CrossEntropyLoss') \
            .build_metric('CategoricalError') \
            .validate_every((1, 'epochs')) \
            .save_every((1, 'epochs'), to_directory=os.path.join(self.ROOT_DIR, 'saves')) \
            .save_at_best_validation_score() \
            .set_max_num_epochs(2)

        # Try to serialize
        trainer.save()

        # Try to unserialize
        trainer = Trainer(net).save_to_directory(os.path.join(self.ROOT_DIR, 'saves')).load()

    @skipUnless(torch.cuda.device_count() >= 4, "Not enough cuda devices for test_multi_gpu.")
    def test_multi_gpu(self):
        if not torch.cuda.is_available():
            return

        from inferno.trainers.basic import Trainer
        from inferno.io.box.cifar import get_cifar10_loaders
        import os

        # Make model
        net = self._make_test_model()
        # Make trainer
        trainer = Trainer(model=net) \
            .build_optimizer('Adam') \
            .build_criterion('CrossEntropyLoss') \
            .build_metric('CategoricalError') \
            .validate_every((1, 'epochs')) \
            .save_every((1, 'epochs'), to_directory=os.path.join(self.ROOT_DIR, 'saves')) \
            .save_at_best_validation_score() \
            .set_max_num_epochs(2)\
            .cuda(devices=[0, 1, 2, 3], base_device='cpu')

        train_loader, validate_loader = get_cifar10_loaders(root_directory=self.ROOT_DIR,
                                                            download=True)
        trainer.bind_loader('train', train_loader)
        trainer.bind_loader('validate', validate_loader)

        trainer.fit()

    def test_save(self):
        from inferno.trainers.basic import Trainer
        trainer = Trainer().save_to_directory(to_directory=self.ROOT_DIR,
                                              checkpoint_filename='dummy.pytorch')
        trainer.save()
        # Instantiate new trainer and load
        trainer = Trainer().load(from_directory=self.ROOT_DIR, filename='dummy.pytorch')

    @skipUnless(torch.cuda.device_count() >= 2, "Not enough cuda devices for test_multi_gpu_setup.")
    def test_multi_gpu_setup(self):
        from torch.nn import CrossEntropyLoss
        from inferno.trainers.basic import Trainer
        # Test base_device = 'cpu'
        # Build model
        net = self._make_test_model()
        # Make dummy criterion
        criterion = CrossEntropyLoss(weight=torch.rand(10))
        # Make trainer
        trainer = Trainer(net).build_criterion(criterion).cuda([0, 1], base_device='cpu')
        self.assertIsInstance(trainer.criterion.weight, torch.FloatTensor)
        # Test base_device = 'cpu'
        # Build model
        net = self._make_test_model()
        criterion = CrossEntropyLoss(weight=torch.rand(10))
        # Make trainer
        trainer = Trainer(net).build_criterion(criterion).cuda([0, 1], base_device='cuda')
        self.assertIsInstance(trainer.criterion.weight, torch.cuda.FloatTensor)


if __name__ == '__main__':
    main()
