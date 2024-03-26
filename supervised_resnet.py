import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import random


epochs = 10
learningRate = 0.001
batchSize = 16


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#1Load data
def load_data(train_data,augment=False):
    """Load CIFAR-10 (training and test set)."""

    augmentedTransform=transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )]
    )

    regularTransform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )]
    )
    if augment == True:
        trainset = CIFAR10(".", train=True, download=True, transform=augmentedTransform)
    else:
        trainset = CIFAR10(".", train=True, download=True, transform=regularTransform)
    testset = CIFAR10(".", train=False, download=True, transform=regularTransform)

    """#Split train into train and dev (split is 80/10)
    proportion = int(len(trainset) /100 *80)
    train, dev = torch.utils.data.random_split(trainset, [proportion,len(trainset)-proportion])

    trainloader = DataLoader(train, batch_size=batchSize, shuffle=True)
    devloader = DataLoader(dev, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(testset, batch_size=batchSize)
    num_examples = {"trainset" : len(trainset), "devset" : len(devloader), "testset" : len(testset)}
    return trainloader, devloader, testloader, num_examples"""

    
    """Load CIFAR10 data."""

    """transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)"""
    """ #Split train into train and dev (split is 80/20)"""
    
    if(train_data):
        proportion = int(len(trainset) /100 *80)
        print(proportion)
        train, dev = torch.utils.data.random_split(trainset, [proportion,len(trainset)-proportion])
    else:
        random.seed(42)
        split_train = int(len(trainset) /100 *33)
        train1, dev1 = torch.utils.data.random_split(trainset, [split_train,len(trainset)-split_train])
        proportion = int(len(train1) /100 *80)
        print(proportion)
        train, dev = torch.utils.data.random_split(train1, [proportion,len(train1)-proportion])
    trainloader = DataLoader(train, batch_size=batchSize, shuffle=True)
    devloader = DataLoader(dev, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(testset, batch_size=batchSize)
    return trainloader,devloader,testloader




#2-Alternatively: Load a pretrained ResNet model and adapt it to our case


def initialize_model():
    # Load pretrained model params
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # We do not want to modify the parameters of ResNet
    for param in model.parameters():
        param.requires_grad = False

    # Replace the original classifier with a new Linear layer
    model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 10)
)
#    num_features = model.fc.in_features
#    model.fc = nn.Linear(num_features, 10)

    return model
net = initialize_model()
net.to(device)

#3 define loss

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#4 train
def train(trainloader,devloader):
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0


        #Calculate loss on validation set
        correct = 0
        total = 0
        val_loss = 0
        patience=3
        best_val_loss = float('inf')
        epochs_no_improve = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in devloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item() * labels.size(0)
        valAccuracy = 100 * correct // total
        val_loss /= len(devloader.dataset)
        print(f'Validation Accuracy : {valAccuracy} %, Validation Loss :{val_loss}')
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping after epoch {epoch + 1}")
                break

    print('Finished Training')


#5 eval
def test(testloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test data = { correct / total} ')



trainloader1, devloader1, testloader1 = load_data(True)
train(trainloader1,devloader1)
test(testloader1)
trainloader2, devloader2, testloader2 = load_data(False)
train(trainloader2,devloader2)
test(testloader2)
