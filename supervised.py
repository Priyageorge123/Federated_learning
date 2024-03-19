import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import wandb


epochs = 10
learningRate = 0.001
batchSize = 16

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

    """#augmentedTransform=transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )]
    )

    #regularTransform = transforms.Compose(
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
    testset = CIFAR10(".", train=False, download=True, transform=regularTransform)"""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root=".", train=True, download=True, transform=transform)
    testset = CIFAR10(root=".", train=False, download=True, transform=transform)
    
    #Slit train into train and dev (split is 90/10)
    proportion = int(len(trainset) /100 *80)
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


class ConvNet(nn.Module):
  def __init__(self, num_classes = 10):
    super(ConvNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 12, kernel_size=3, stride = 1, padding = 1)
    #Shape = (32, 12, 224, 224)
    self.bn1 = nn.BatchNorm2d(num_features = 12)
    #Shape = (32, 12, 224, 224)
    self.relu1 = nn.ReLU()
    #Shape = (32, 12, 224, 224)
    self.pool = nn.MaxPool2d(kernel_size = 2)
    #Reduce the image size be factor 2
    #Shape = (32, 12, 112, 112)
    self.conv2 = nn.Conv2d(in_channels = 12, out_channels = 20, kernel_size = 3, stride = 1, padding = 1)
    #Shape = (32, 20, 112, 112)
    self.relu2 = nn.ReLU()
    #Shape = (32, 20, 112, 112)
    self.conv3 = nn.Conv2d(in_channels = 20, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
    self.bn3 = nn.BatchNorm2d(num_features=32)
    self.relu3 = nn.ReLU()
    #Shape = (32, 32, 112, 112)
    self.fc1 = nn.Linear(in_features = 112*112*32, out_features = 120)
    self.fc2 = nn.Linear(in_features = 120, out_features = num_classes)

  def forward(self, input):
    output = self.conv1(input)
    output = self.bn1(output)
    output = self.relu1(output)
    output = self.pool(output)
    output = self.conv2(output)
    output = self.relu2(output)
    output = self.conv3(output)
    output = self.bn3(output)
    output = self.relu3(output)
    #Above output will be in matrix form, with shape (32, 32, 112, 112)
    output = output.view(-1, 112*112*32)
    output = F.relu(self.fc1(output))
    output = self.fc2(output)
    return output

#net = ConvNet()
#net.to(device)


#2-Alternatively: Load a pretrained ResNet model and adapt it to our case

from torchvision import models

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
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)
#    num_features = model.fc.in_features
#    model.fc = nn.Linear(num_features, 10)

    return model
net = initialize_model()
net.to(device)

#3 define loss

#criterion = nn.NLLLoss()
#optimizer = optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
