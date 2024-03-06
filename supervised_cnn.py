"""
This is a simple baseline, which trains a model similar to federatedClient

"""

import argparse
import warnings
from collections import OrderedDict

from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batchSize = 16


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def load_data():
    """Load CIFAR10 data."""
    transform = transforms.Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    """ #Split train into train and dev (split is 80/20)"""
    proportion = int(len(trainset) /100 *80)
    train, dev = torch.utils.data.random_split(trainset, [proportion,len(trainset)-proportion])
    trainloader = DataLoader(train, batch_size=batchSize, shuffle=True)
    devloader = DataLoader(dev, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(testset, batch_size=batchSize)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(epochs):
        for batch in tqdm(trainloader, "Training"):
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

        # Added a evaluation loop
        loss, acc = test(net, devloader)
        print("Epoch = " +str(epoch) +"Accuracy= " + str(acc))


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader, "Testing"):
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)

    return loss, accuracy


net = Net().to(DEVICE)
trainloader, devloader, testloader = load_data()

train(net, trainloader, epochs=3)


loss, acc = test(net, testloader)
print("Accuracy on the test data= " + str(acc))