import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import wandb


epochs = 10
learningRate = 0.001
batchSize = 4

keyFile = open('wandb.key', 'r')
WANDB_API_KEY = keyFile.readline().rstrip()
wandb.login(key=WANDB_API_KEY)
wandb.init(
    # set the wandb project where this run will be logged
    project="CIFAR-supervised",

    # track hyperparameters and run metadata
    config={
        "architecture": "ResNet",
        "dataset": "CIFAR-10",
        "learning_rate": learningRate,
        "epochs": epochs,
        "batchSize" : batchSize
    }
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#1Load data
def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    #Slit train into train and dev (split is 90/10)
    proportion = int(len(trainset) /100 *90)
    train, dev = torch.utils.data.random_split(trainset, [proportion,len(trainset)-proportion])

    trainloader = DataLoader(train, batch_size=batchSize, shuffle=True)
    devloader = DataLoader(dev, batch_size=batchSize, shuffle=True)
    testloader = DataLoader(testset, batch_size=batchSize)
    num_examples = {"trainset" : len(trainset), "devset" : len(devloader), "testset" : len(testset)}
    return trainloader, devloader, testloader, num_examples


trainloader, devloader, testloader, num_examples = load_data()


#2. define NN manually
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.to(device)


#2-Alternatively: Load a pretrained ResNet model and adapt it to our case

from torchvision import models

def initialize_model():
    # Load pretrained model params
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Replace the original classifier with a new Linear layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)

    # Ensure all params get updated during finetuning
    for param in model.parameters():
        param.requires_grad = True
    return model
net = initialize_model()
net.to(device)

#3 define loss
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learningRate, momentum=0.9)

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


    #Calculate loss on test set
    correct = 0
    total = 0
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
    valAccuracy = 100 * correct // total
    wandb.log({"acc": valAccuracy})

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