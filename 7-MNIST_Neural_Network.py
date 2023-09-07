import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary


# MNIST
def mnist(batch_sz, valid_size=0.2, shuffle=True, random_seed=2000):
    num_classes = 10
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ]
    )

    transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Training dataset
    train_data = MNIST(
        root="./datasets", train=True, download=True, transform=transform_train
    )
    valid_data = MNIST(
        root="./datasets", train=True, download=True, transform=transform_valid
    )
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_sz, sampler=train_sampler, pin_memory=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_sz, sampler=valid_sampler, pin_memory=True
    )

    # Test dataset
    test_data = MNIST(
        root="./datasets", train=False, download=True, transform=transform_test
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_sz, shuffle=False, pin_memory=True
    )

    return train_loader, valid_loader, test_loader


batch_sz = 64  # this is batch size i.e. the number of rows in a batch of data
train_loader, valid_loader, test_loader = mnist(batch_sz)

batch = next(iter(train_loader))

batch[0].shape

plt.imshow(batch[0][52, 0, :, :], cmap="gray")
plt.title(batch[1][52])


class My_NN(nn.Module):
    def __init__(self, ni, nh, no):
        super().__init__()

        self.layer1 = nn.Linear(ni, nh)
        # define the second layer from (nh, no)
        self.layer2 = nn.Linear(nh, no)
        self.ni = ni

    def forward(self, x):
        # reshape x to (batch_size, feature)
        feature = self.ni
        x = torch.reshape(x, (x.shape[0], feature))

        x = self.layer1(x)
        x = torch.sigmoid(x)

        # forward the second layer
        x = self.layer2(x)
        # x = torch.sigmoid(x)

        return x


device = torch.device("cuda:0")
ni = 28 * 28
nh = 100
no = 10

net = My_NN(ni, nh, no).to(device)
num_epochs = 10
learning_rate = 2
opt = optim.SGD(net.parameters(), lr=learning_rate)
ls = []
for i in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        X, y = batch[0].to(device), batch[1].to(device)
        yhat = net.forward(X)
        loss = F.cross_entropy(yhat, y)
        total_loss += loss.item()
        opt.zero_grad()
        # Backpropagation loss
        loss.backward()
        opt.step()

    ls.append(total_loss / len(train_loader))

plt.plot(ls)


device = torch.device("cuda:0")
ys = []
yhats = []

with torch.no_grad():
    for batch in test_loader:
        X, y = batch[0].to(device), batch[1].to(device)
        net.to(device)
        yhat = net(X)
        labels = torch.argmax(yhat, axis=1)
        ys.extend(y.cpu().numpy())
        yhats.extend(labels.cpu().numpy())

ys = np.array(ys)
yhats = np.array(yhats)
print(ys)
print(yhats)

yhats.shape

# compute the test accuracy
# step 1: get the number of test examples
num_all_test_examples = len(ys)

# step2: compute how many samples are correctly classifieed
num_correct = 0
for i in range(len(ys)):
    if ys[i] == yhats[i]:
        num_correct += 1

# step compute the test accuracy
print(num_correct / num_all_test_examples)

net = My_NN(784, 100, 10)
for params in net.parameters():
    print(params.shape)

for batch in test_loader:
    print(batch)
    break

print(batch[0].shape)
print(batch[1])

plt.figure(figsize=(16, 16))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    plt.imshow(batch[0][i, 0, :, :])
    plt.title(batch[1][i].item())
    plt.axis("off")


class My_NN(nn.Module):
    def __init__(self, ni, nh1, nh2, nh3, nh4, no):
        super().__init__()
        self.ni = ni
        self.linear1 = nn.Linear(ni, nh1)
        self.linear2 = nn.Linear(nh1, nh2)
        self.linear3 = nn.Linear(nh2, nh3)
        self.linear4 = nn.Linear(nh3, nh4)
        self.linear5 = nn.Linear(nh4, no)

    def forward(self, x):
        x = x.view(-1, self.ni)
        x = self.linear1(x)
        x = torch.sigmoid(x)

        x = self.linear2(x)
        x = torch.sigmoid(x)

        x = self.linear3(x)
        x = torch.sigmoid(x)

        x = self.linear4(x)
        x = torch.sigmoid(x)

        x = self.linear5(x)
        return x


device = torch.device("cuda:0")

ni = 28 * 28
nh1 = 128
nh2 = 128
nh3 = 64
nh4 = 64
no = 10

net = My_NN(ni, nh1, nh2, nh3, nh4, no).to(device)
lr = 2
opt = optim.SGD(net.parameters(), lr=lr)
num_epochs = 10

ls = []
vls = []
for i in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # getting data and labels from the batch
        X, y = batch[0].to(device), batch[1].to(device)
        # doing forward pass through the network
        yhat = net.forward(X)
        # calculating loss
        loss = F.cross_entropy(yhat, y)
        # emptyting hte optimizer buffes to store gradients
        opt.zero_grad()
        # calculating gradients
        # taking a step in the negative gradient direction
        loss.backward()
        opt.step()
        total_loss += loss.item()
    with torch.no_grad():
        val_loss = 0
        for batch in valid_loader:
            X, y = X, y = batch[0].to(device), batch[1].to(device)
            yhat = net.forward(X)
            # calculating loss
            loss = F.cross_entropy(yhat, y)
            val_loss += loss.item()

    ls.append(total_loss / len(train_loader))
    vls.append(val_loss / len(valid_loader))


plt.plot(ls, "b")
plt.plot(vls, "g")
