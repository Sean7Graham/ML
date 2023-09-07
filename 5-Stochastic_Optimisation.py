import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models,transforms
from torchvision.datasets import MNIST
from torch.utils.data.sampler import SubsetRandomSampler

# GENERATE MNIST DATASET - no need to edit
def mnist(batch_sz, valid_size=0.2, shuffle=True, random_seed=2000):
    num_classes = 10
    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                    ])
    
    transform_valid = transforms.Compose([
                        transforms.ToTensor(),
                    ])
    
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                    ])
    
    # Training dataset
    train_data = MNIST(root='./datasets', train=True, download=True, transform=transform_train)
    valid_data = MNIST(root='./datasets', train=True, download=True, transform=transform_valid)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_sz, sampler=train_sampler,pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_sz, sampler=valid_sampler,pin_memory=True)

    # Test dataset
    test_data = MNIST(root='./datasets', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_sz, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader

batch_size = 64 # this is batch size i.e. the number of rows in a batch of data, feel free try other values {32, 64, 128, 256, 512}

input_dimension = 1 * 28 * 28 # MNIST images are grey scale images 28 pixels wide and high 
d = input_dimension # if you prefer a shorter varible name

number_of_classes = 10 # number of class of MNIST data set C = {0,1,2,3,4,5,6,7,8,9}

train_loader, valid_loader, test_loader=mnist(batch_size) # Create the data loaders these are objects that when looping over present batchs of data

# lists to store loss and accuracy values
sgd_train_losses, sgd_val_losses, sgd_train_acc, sgd_val_acc = [], [], [], []
opt_train_losses, opt_val_losses, opt_train_acc, opt_val_acc = [], [], [], []
adam_train_losses, adam_val_losses, adam_train_acc, adam_val_acc = [], [], [], []

batch = next(iter(train_loader))
batch_content = batch[0]
print('dimension of single batch: ',batch_content.size())
print('code will convert 28*28 image in a single 784 vector of pixel values on which we will fit a linear model')

# Visulise some training examples
k = 0
h = 10
f, axs = plt.subplots(1,h)

for k in range(h):
  img = batch_content[k,:,:,:].squeeze()
  axs[k].imshow(img)
  axs[k].axis("off")
  
# By default the data loader gives you the class value as a integer rather than a one hot vector as show above

def index_to_one_hot(y, classes=10):
  b = len(y) # batchsize
  one_hot = torch.zeros((b, classes), device=y.device)
  xs = torch.arange(b, device=y.device)
  one_hot[xs,y] = 1
  return one_hot

y = torch.randint(10,(10,))
print('y',y)
print('y',index_to_one_hot(y))

# As before please complete the following two functions. Remember here X and y are batches of b examles.

def CE_loss(X, ystar, w):
  m = torch.matmul(X,w)
  softmax_probs = torch.softmax(m, dim=1)
  y_indices = torch.argmax(ystar,dim=1, keepdim = True)
  log_probs = torch.log(softmax_probs.gather(1, y_indices))
  loss = -torch.mean(log_probs)

  return float(loss)

Y = index_to_one_hot(y) 
X = torch.randn(10,10)

print('your function:',float(CE_loss(X,Y,torch.eye(10).float())))
print('should match this function:',float(F.cross_entropy(X, y)))

def grad_func(X, ystar, w):
  n = X.size()[0]
  grad = -1/n *X.T@(ystar - F.softmax(X@w, dim = 1))
  return grad

# useful function to calculate accuracy

def accuracy(out, targets):
    _, pred = torch.max(out, 1)
    correct = torch.eq(pred, targets).sum()
    acc = torch.mean(torch.eq(pred, targets).float())
    return correct, float(100. * acc)

def test(w, valid_loader): # function that evaluates loss on vaildation set
    total_loss, total_correct, total_num = 0, 0, 0 
    for batch in valid_loader:
      X,y = batch[0].to(device).view(batch[0].size()[0],input_dimension), batch[1].to(device)
      Y = index_to_one_hot(y, classes=10)
      loss_train = CE_loss(X, Y, w)
      correct, acc = accuracy(X.mm(w), y)
      total_correct += correct
      total_loss += loss_train*X.size()[0]
      total_num += X.size()[0]
    return float(total_loss/total_num), float(total_correct/total_num * 100)

with torch.no_grad(): # in this exersize we will not use autograd you will have calucated gradent

  # fixing a device to run the codes on
  device = torch.device("cuda:0")

  w = torch.zeros(input_dimension,number_of_classes).to(device)
  m = torch.zeros(input_dimension,number_of_classes).to(device)
  
  eta = 1e-1
  num_epochs = 10
  mu = 0.8
  m = 0

  for i in range(num_epochs):
    total_loss, total_correct, total_num = 0, 0, 0 
    for batch in train_loader:
      #extractign data and labels from batch
      X,y = batch[0].to(device).view(batch[0].size()[0],input_dimension), batch[1].to(device)
      Y = index_to_one_hot(y, classes=10)      
      # ----------------------------
      # Optimisation -  Your code here 
      m = m*mu - eta*grad_func(X, Y, w)
      w = w + m
      # ----------------------------
      # track loss and accuracy after every step
      loss_train = CE_loss(X, Y, w)
      correct, acc = accuracy(X.mm(w), y)
      total_correct += correct
      total_loss += loss_train*X.size()[0]
      total_num += X.size()[0]

    # ----------------------------
    # After each epoch track metrics
    sgd_train_losses.append(float(total_loss/total_num))
    sgd_train_acc.append(float(total_correct/total_num * 100))
    val_loss, val_acc = test(w, valid_loader)
    sgd_val_losses.append(val_loss)
    sgd_val_acc.append(val_acc)
    print('train loss: {:.2e} | train acc: {:.2f} | eta: {:.2f}'.format(float(total_loss/total_num), float(total_correct/total_num * 100), eta))
    print('val loss: {:.2e} | val acc: {:.2f}'.format(val_loss, val_acc))
    print('-'*50)
    # ----------------------------
    # Adjust the learning rate
    eta *= 0.9
    
    # plot stuff - reuse this cell for plotting bonus material too.
f, axs = plt.subplots(2,2)
axs[0,0].plot(np.arange(len(sgd_train_losses)),sgd_train_losses, 'red')
axs[0,0].plot(np.arange(len(opt_train_losses)),opt_train_losses, 'blue')
axs[0,0].plot(np.arange(len(adam_train_losses)),adam_train_losses, 'black')
axs[0,0].set_ylabel('train losses')
axs[0,1].plot(np.arange(len(sgd_train_acc)),sgd_train_acc, 'red')
axs[0,1].plot(np.arange(len(opt_train_acc)),opt_train_acc, 'blue')
axs[0,1].plot(np.arange(len(adam_train_acc)),adam_train_acc, 'black')
axs[0,1].set_ylabel('train acc')
axs[1,0].plot(np.arange(len(sgd_val_losses)),sgd_val_losses, 'red')
axs[1,0].plot(np.arange(len(opt_val_losses)),opt_val_losses, 'blue')
axs[1,0].plot(np.arange(len(adam_val_losses)),adam_val_losses, 'black')
axs[1,0].set_ylabel('val losses')
axs[1,1].plot(np.arange(len(sgd_val_acc)),sgd_val_acc, 'red')
axs[1,1].plot(np.arange(len(opt_val_acc)),opt_val_acc, 'blue')
axs[1,1].plot(np.arange(len(adam_val_acc)),adam_val_acc, 'black')
axs[1,1].set_ylabel('val acc')

# We can visualise the learnt weights (one for each class) as images by reshapeing into 28x28 grids

w_cpu = w.cpu().clone()
print(w_cpu.size())

f, axs = plt.subplots(1,number_of_classes)

for k in range(h):
  img = w_cpu[:,k].view(28,28)
  axs[k].imshow(img)
  axs[k].set_title(k)
  axs[k].axis("off")
  
 with torch.no_grad(): # in this exersize we will not use autograd you will have calucated gradent

  # fixing a device to run the codes on
  device = torch.device("cuda:0")

  w = torch.zeros(input_dimension,number_of_classes).to(device)
  m = torch.zeros(input_dimension,number_of_classes).to(device)
  
  eta = 1e-1
  num_epochs = 10

  for i in range(num_epochs):
    total_loss, total_correct, total_num = 0, 0, 0 
    for batch in train_loader:
      #extractign data and labels from batch
      X,y = batch[0].to(device).view(batch[0].size()[0],input_dimension), batch[1].to(device)
      Y = index_to_one_hot(y, classes=10)      
      # ----------------------------
      # Optimisation TODO




      # ----------------------------
      # track loss and accuracy after every step
      loss_train = CE_loss(X, Y, w)
      correct, acc = accuracy(X.mm(w), y)
      total_correct += correct
      total_loss += loss_train*X.size()[0]
      total_num += X.size()[0]

    # ----------------------------
    # After each epoch track metrics
    opt_train_losses.append(float(total_loss/total_num))
    opt_train_acc.append(float(total_correct/total_num * 100))
    val_loss, val_acc = test(w, valid_loader)
    opt_val_losses.append(val_loss)
    opt_val_acc.append(val_acc)
    print('train loss: {:.2e} | train acc: {:.2f} | eta: {:.2f}'.format(float(total_loss/total_num), float(total_correct/total_num * 100), eta))
    print('val loss: {:.2e} | val acc: {:.2f}'.format(val_loss, val_acc))
    print('-'*50)
    # ----------------------------
    # Adjust the learning rate
    eta *= 0.9 
    
with torch.no_grad(): # in this exersize we will not use autograd you will have calucated gradent

  # fixing a device to run the codes on
  device = torch.device("cuda:0")

  w = torch.zeros(input_dimension,number_of_classes).to(device)
  m = torch.zeros(input_dimension,number_of_classes).to(device)
  
  eta = 1e-3
  beta_1 = 0.9
  beta_2 = 0.999
  num_epochs = 10

  for i in range(num_epochs):
    total_loss, total_correct, total_num = 0, 0, 0 
    for batch in train_loader:
      #extractign data and labels from batch
      X,y = batch[0].to(device).view(batch[0].size()[0],input_dimension), batch[1].to(device)
      Y = index_to_one_hot(y, classes=10)      
      # ----------------------------
      # Optimisation -  TODO 




      # ----------------------------
      # track loss and accuracy after every step
      loss_train = CE_loss(X, Y, w)
      correct, acc = accuracy(X.mm(w), y)
      total_correct += correct
      total_loss += loss_train*X.size()[0]
      total_num += X.size()[0]

    # ----------------------------
    # After each epoch track metrics
    adam_train_losses.append(float(total_loss/total_num))
    adam_train_acc.append(float(total_correct/total_num * 100))
    val_loss, val_acc = test(w, valid_loader)
    adam_val_losses.append(val_loss)
    adam_val_acc.append(val_acc)
    print('train loss: {:.2e} | train acc: {:.2f} | eta: {:.2f}'.format(float(total_loss/total_num), float(total_correct/total_num * 100), eta))
    print('val loss: {:.2e} | val acc: {:.2f}'.format(val_loss, val_acc))
    print('-'*50)
