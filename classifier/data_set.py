import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

batch_size = 4

training_set = torchvision.datasets.CIFAR10(
        root = "data", train = True,
        download=True, transform=transform
)

testing_set = torchvision.datasets.CIFAR10(
        root = "data", train = False,
        download=True, transform=transform
)

training_loader = DataLoader(
        training_set, batch_size=batch_size,
        shuffle=True, num_workers=2
)

testing_loader = DataLoader(
        testing_set, batch_size=batch_size,
        shuffle=False, num_workers=2
)

classes = (
    'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
    )





