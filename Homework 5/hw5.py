# hw5.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

#----------------------------------
#           Problem 1
#----------------------------------

'''
# inherit from nn.Module
class AlexNet_CNN(nn.Module):
    def __init__(self):
        # initialize from nn.Module
        super(AlexNet_CNN, self).__init__()

        # initialize each layer
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU())
        self.layer_2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=1),
            nn.ReLU())
        self.layer_3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer_4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer_5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=4, padding=0),
            nn.ReLU())
        
    # forward propogation function
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)

Alex_Net = AlexNet_CNN()
print(summary(Alex_Net, (3, 244, 244)))
'''

#----------------------------------
#           Problem 2
#----------------------------------

# load mnist data 70,000 samples
mnist = loadmat('input/mnist-original.mat')
mnist_data = np.array(mnist['data']).T
mnist_label = np.array(mnist['label']).T

# use one hot encoding to get each image
#ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(mnist_label)
# print(f'\nEncoded categories: ',ohe.categories_)
#mnist_label = ohe.transform(mnist_label)
# print(f'\nmnist_labels encoded: ',mnist_label)

# reshape the input data into image shape
mnist_data = mnist_data.reshape((70000, 1, 28, 28))

# get the training data into a tensor
mnist_data = torch.tensor(mnist_data, dtype=torch.float32)
mnist_label = torch.tensor(mnist_label, dtype=torch.float32).squeeze().type(torch.LongTensor)

# separate test and train data
X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_label, train_size=0.8, random_state=7)

# separate validation data out too (0.8*0.75 = 0.6)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.75, random_state=7)

# image shape
print(f'\nX_train shape:', X_train.shape)

# show image
plt.imshow(X_train[0][0],cmap='Greys')
plt.savefig('output/image_visualization.png')

# build the CNN
#class Mnist_CNN_ReLU(nn.Module):
class Mnist_CNN_Sigmoid(nn.Module):
    def __init__(self):
        # initialize from nn.Module
        super().__init__()

        # initialize each layer according to HW5
        # convolutional layers
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=(3,3), stride=1, padding=1),
            #nn.ReLU(),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2,2), stride=2))
        self.layer_2 = nn.Sequential(
            nn.Conv2d(28, 56, kernel_size=(3,3), stride=1, padding=1),
            #nn.ReLU(),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=(2,2), stride=1))
                
        # dense layers
        self.layer_3 = nn.Sequential(
            nn.Linear(9464, 128),
            nn.Sigmoid())
            #nn.ReLU())
        self.layer_4 = nn.Sequential(
            nn.Linear(128, 10),
            nn.LogSoftmax(1))
        
    # forward propogation function
    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = x.view(x.size(0), -1)
        x = self.layer_3(x)
        x = self.layer_4(x)
        return x
    
# create the model, loss, and optimizer functions
#ReLU_model = Mnist_CNN_ReLU()
Sigmoid_model = Mnist_CNN_Sigmoid()
loss_fn = nn.CrossEntropyLoss()
Sigmoid_optimizer = optim.SGD(Sigmoid_model.parameters(), lr=0.001)

# do stochastic gradient descent on given epochs
epoch = 5
batch_size = 5

# keep track of loss
epoch_train_loss = []
epoch_val_loss = []

for n in range(epoch):
    # train the model
    Sigmoid_model.train()
    for i in range(0, len(X_train), batch_size):
        # do ReLU optimization
        y_pred_Sigmoid = Sigmoid_model(X_train[i:i+batch_size])
        loss_Sigmoid = loss_fn(y_pred_Sigmoid, y_train[i:i+batch_size])
        Sigmoid_optimizer.zero_grad()
        loss_Sigmoid.backward()
        Sigmoid_optimizer.step()
        epoch_train_loss.append(loss_Sigmoid.item())

    # validation loss
    Sigmoid_model.eval()
    y_pred_Sigmoid_val = Sigmoid_model(X_val)
    epoch_val_loss.append(loss_fn(y_pred_Sigmoid_val, y_val).item())

    # testing accuracy

    y_pred = Sigmoid_model(X_test)
    acc = (torch.argmax(y_pred, 1) == y_test).float().sum()
    count = len(y_test)
    acc /= count
    print(f'Sigmoid accuracy of epoch ',n,':',acc.item()*100)

# plot training loss
plt.figure(figsize=(10,6))
plt.title("Sigmoid Training Loss per Batch")
plt.plot(epoch_train_loss)
plt.xlim(0, 500)
plt.ylim(0, 100)
plt.savefig('output/Sigmoid_Training_Loss.png')

plt.clf()
plt.figure(figsize=(10,6))
plt.title("Sigmoid Validation Loss per Epoch")
plt.plot(epoch_val_loss)
plt.xlim(0, 5)
plt.ylim(0, 100)
plt.savefig('output/Sigmoid_Validation_Loss.png')