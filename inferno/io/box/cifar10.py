import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(root_directory, train_batch_size=128, test_batch_size=100,
                        download=False):
    # Data preparation for CIFAR10. Borrowed from
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

    trainset = torchvision.datasets.CIFAR10(root=os.path.join(root_directory, 'data'),
                                            train=True, download=download,
                                            transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=os.path.join(root_directory, 'data'),
                                           train=False, download=download,
                                           transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader