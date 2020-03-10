import os, sys

import torch

from torchvision import transforms
from torchvision import datasets


def get_data_loader():
    # Train Phase transformations
    train_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                        # Note the difference between (0.1307) and (0.1307,)
                                        ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
                                        #  transforms.Resize((28, 28)),
                                        #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

    train_dataset = datasets.MNIST('./data',
                                train=True,
                                transform=train_transforms,
                                download=True)

    test_dataset = datasets.MNIST('./data',
                                train=False,
                                transform=test_transforms,
                                download=True)
    SEED = 1
    torch.manual_seed(SEED)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(SEED)

    dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

    return train_loader, test_loader