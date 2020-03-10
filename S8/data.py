import os, sys

import torch

import torchvision
from torchvision import transforms
from torchvision import datasets


def get_data_loader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    SEED = 1
    torch.manual_seed(SEED)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(SEED)

    dataloader_args = dict(num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=8,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes