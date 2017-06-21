from unittest import TestCase


class TestTrainer(TestCase):
    # Parameters
    ROOT_DIR = None

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
        resnet = torchvision.models.resnet18()

        # Make trainer
        trainer = Trainer(model=resnet)\
            .build_logger(log_directory=os.path.join(self.ROOT_DIR, 'logs'))\
            .build_optimizer('Adam')\
            .build_criterion('CrossEntropyLoss')\
            .build_metric('CategoricalAccuracy')\
            .validate_every((1, 'epochs'))\
            .save_every((10, 'epochs'), to_directory=os.path.join(self.ROOT_DIR, 'saves'))\
            .save_at_best_validation_score()\
            .set_max_num_epochs(200)\

        # Bind trainer to datasets and fit
        trainer.bind_loader('train', trainloader).bind_loader('validate', testloader).fit()
