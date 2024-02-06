
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
import wandb
import os

# Initialize wandb
keyFile = open('wandb.key', 'r')
WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)

epochs = 10
learningRate = 0.001
batchSize = 16

# Define wandb initialization
wandb.init(
    project="CIFAR-supervised",
    config={
        "architecture": "ResNet",
        "dataset": "CIFAR-10",
        "learning_rate": learningRate,
        "epochs": epochs,
        "batchSize": batchSize,
        "augment": False  # Set augment flag accordingly
    }
)



# Sweep configuration
sweep_config = {
    "method": "random",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "batchSize": {"values": [4, 8, 16, 32]},
        "learning_rate": {"values": [0.01,0.001,0.0001]},
        "augment": {"values": [True, False]}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="CIFAR-supervised")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(config):
    augment=config.augment
    # Define transforms
    regularTransform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    # Define augmentation transform

    augmentedTransform = transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    # Load CIFAR-10 dataset
    if augment:
      trainset = CIFAR10(".", train=True, download=True, transform=augmentedTransform)
    else:
      trainset = CIFAR10(".", train=True, download=True, transform=regularTransform)
    testset = CIFAR10(".", train=False, download=True, transform=regularTransform)

    # Split train into train and dev (split is 90/10)
    proportion = int(len(trainset) / 100 * 90)
    train, dev = torch.utils.data.random_split(trainset, [proportion, len(trainset) - proportion])

    # Create data loaders
    trainloader = DataLoader(train, batch_size=config.batchSize, shuffle=True)
    devloader = DataLoader(dev, batch_size=config.batchSize, shuffle=True)
    testloader = DataLoader(testset, batch_size=config.batchSize)

    num_examples = {"trainset": len(trainset), "devset": len(devloader), "testset": len(testset)}
    return trainloader, devloader, testloader, num_examples

def initialize_model():
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    )

    return model

def train():
    config = wandb.config
    trainloader, devloader, testloader, num_examples = load_data(config)

    # Initialize model
    net = initialize_model()
    net.to(device)

    # Loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()

            # Print average loss per epoch
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    
        #Calculate loss on validation set
        correct = 0
        total = 0
        val_loss = 0
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
        wandb.log({"val_acc": valAccuracy, "val_loss" :  val_loss} )
        print(f'Validation Accuracy : {valAccuracy} %, Validation Loss :{val_loss}')
    
    print('Finished Training')
    
    
    #5 eval
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
    
    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    
    wandb.finish()

# Start the sweep
wandb.agent(sweep_id, train, count=5)
