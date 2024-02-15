import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import wandb
import torch.nn as nn
from torchvision import models

epochs = 10


keyFile = open('wandb.key', 'r')
WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)


# Sweep configuration
sweepName="federated-sweep"
sweep_config = {
    "method": "random",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "parameters": {
        "batchSize": {"values": [4, 8, 16, 32, 64, 128]},
        "learning_rate": {"min" : 0.00001, "max" : 0.01},
        "augment": {"values": [True, False]},

        "dropout_rate": {"values" : [0.1, 0.3, 0.5] },
        "trainAllParam": {"values": [True, False]},
        "latentSpace": {"values": [32, 64, 128, 256]},

        "modelarchitecture" : {"values" : ["ResNet50", "EfficientNet_V2"]}
    }
}

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def main():
    wandb.init(project=sweepName)
    # 1Load data
    def load_data(augment=wandb.config.augment):
        """Load CIFAR-10 (training and test set)."""

        augmentedTransform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        regularTransform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        if augment == True:
            trainset = CIFAR10(".", train=True, download=True, transform=augmentedTransform)
        else:
            trainset = CIFAR10(".", train=True, download=True, transform=regularTransform)
        testset = CIFAR10(".", train=False, download=True, transform=regularTransform)

        # Slit train into train and dev (split is 90/10)
        proportion = int(len(trainset) / 100 * 90)
        train, dev = torch.utils.data.random_split(trainset, [proportion, len(trainset) - proportion])

        trainloader = DataLoader(train, batch_size=wandb.config.batchSize, shuffle=True)
        devloader = DataLoader(dev, batch_size=wandb.config.batchSize, shuffle=True)
        testloader = DataLoader(testset, batch_size=wandb.config.batchSize)
        num_examples = {"trainset": len(trainset), "devset": len(devloader), "testset": len(testset)}
        return trainloader, devloader, testloader, num_examples

    trainloader, devloader, testloader, num_examples = load_data()

    # 2. define NN manually
    def initialize_model():
        if wandb.config.modelarchitecture == 'ResNet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif wandb.config.modelarchitecture == 'EfficientNet_V2':
            model = models.efficientnet_v2_s(weights = 'EfficientNet_V2_S_Weights.DEFAULT')


        # We do not want to modify the parameters of ResNet
        if wandb.config.trainAllParam == False:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the original classifier with a new Linear layer
        model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, wandb.config.latentSpace),
        nn.ReLU(),
        nn.Dropout(wandb.config.dropout_rate),
        nn.Linear(wandb.config.latentSpace, 10),
        nn.LogSoftmax(dim=1) # For using NLLLoss()
    )
    #    num_features = model.fc.in_features
    #    model.fc = nn.Linear(num_features, 10)

        return model
    net = initialize_model()
    net.to(device)

    #3 define loss

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=wandb.config.learning_rate)

    #4 train
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


# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=sweepName)
wandb.agent(sweep_id, function=main, count=100)