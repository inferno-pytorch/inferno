from unittest import TestCase
from unittest import main
import time


class TestTrainer(TestCase):
    # Parameters
    ROOT_DIR = __file__
    CUDA = True
    HALF_PRECISION = True

    @staticmethod
    def _make_test_model():
        import torch.nn as nn
        from inferno.extensions.layers.reshape import AsMatrix

        toy_net = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                                nn.MaxPool2d(2),
                                nn.Conv2d(128, 128, 3, 1, 1),
                                nn.MaxPool2d(2),
                                nn.Conv2d(128, 256, 3, 1, 1),
                                nn.AdaptiveMaxPool2d((1, 1)),
                                AsMatrix(),
                                nn.Linear(256, 10))
        return toy_net

    def test_cifar(self):
        from inferno.trainers.basic import Trainer
        import torch
        import torchvision
        from torchvision import transforms
        import os

        # Data preparation. Borrowed from
        # https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root=os.path.join(self.ROOT_DIR, 'data'),
                                                train=True, download=False,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True,
                                                  num_workers=2)

        testset = torchvision.datasets.CIFAR10(root=os.path.join(self.ROOT_DIR, 'data'),
                                               train=False, download=False,
                                               transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                 num_workers=2)

        # Make model
        resnet = self._make_test_model()

        tic = time.time()

        # Make trainer
        trainer = Trainer(model=resnet)\
            .build_logger(log_directory=os.path.join(self.ROOT_DIR, 'logs'))\
            .build_optimizer('Adam')\
            .build_criterion('CrossEntropyLoss')\
            .build_metric('CategoricalError')\
            .validate_every((1, 'epochs'))\
            .save_every((1, 'epochs'), to_directory=os.path.join(self.ROOT_DIR, 'saves'))\
            .save_at_best_validation_score()\
            .set_max_num_epochs(2)\

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


if __name__ == '__main__':
    tester = TestTrainer()
    tester.ROOT_DIR = '/export/home/nrahaman/Python/Repositories/inferno/tests/training/root'
    tester.test_cifar()