import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
        "architecture": "CNN",
        "dataset": "CIFAR-10",
        "learning_rate": learningRate,
        "epochs": epochs,
        "batchSize" : batchSize
    }
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#1Load data
def load_data(augment=False):
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
net = Net()
net.to(device)

#3 define loss

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters())

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
