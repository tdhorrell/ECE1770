# ECE 1770 Homework 4

import numpy as np
from sigmoid import sigmoid
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn

#----------------------------------
#           Problem 1
#----------------------------------

'''
# build a neural network to implement the XOR function
# x0'x1 + x0x1'
# can also be abstracted to AND(OR(x0,x1), NAND(x0,x1)), where they are both OR'd but not both present

# define XOR function
def XOR(x0, x1):

    # set weights for each layer
    weights_layer_1 = np.array([[-10, 20, 20], 
                       [20, -15, -15]])
    
    weights_layer_2 = np.array([[-20, 15, 15]])

    # define input as vertical array
    layer_1_input = np.array([[1],
                    [x0],
                    [x1]])
    
    # use matrix multiplication to get weights
    layer_2_input = np.insert(sigmoid(weights_layer_1.dot(layer_1_input)), [0], [1]).reshape(3,1)

    # complete final layer
    layer_3_output = sigmoid(weights_layer_2.dot(layer_2_input))

    # return XOR
    return int(np.round(layer_3_output)[0][0])

print(f'XOR(0,0):',XOR(0,0))
print(f'XOR(1,0):',XOR(1,0))
print(f'XOR(0,1):',XOR(0,1))
print(f'XOR(1,1):',XOR(1,1))
'''

#----------------------------------
#           Problem 2
#----------------------------------

# load mnist data 70,000 samples
mnist = loadmat('input/mnist-original.mat')
mnist_data = np.array(mnist['data']).T
mnist_label = np.array(mnist['label']).T

# use one hot encoding to get each image
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(mnist_label)
# print(f'\nEncoded categories: ',ohe.categories_)
mnist_label = ohe.transform(mnist_label)
# print(f'\nmnist_labels encoded: ',mnist_label)

# create parameters to loop through
hidden_layer = [16, 32, 64]
epoch_count = [50, 100, 200]

# get the training data into a tensor
mnist_data = torch.tensor(mnist_data, dtype=torch.float32)
mnist_label = torch.tensor(mnist_label, dtype=torch.float32)

# separate test and train data
X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_label, train_size=0.7, random_state=7)

# get shape of the tensors
# print(f'\nX_train shape:', X_train.shape)

# loop through each layer size
for layer_size in hidden_layer:
    # change the number of epochs
    for epoch in epoch_count:
        # create the models
        ReLU_model = nn.Sequential(
            nn.Linear(784, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 10),
        )

        Sigmoid_model = nn.Sequential(
            nn.Linear(784, layer_size),
            nn.Sigmoid(),
            nn.Linear(layer_size, 10),
        )

        # define a loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        ReLU_optimizer = torch.optim.Adam(ReLU_model.parameters(), lr = 0.001)
        Sigmoid_optimizer = torch.optim.Adam(Sigmoid_model.parameters(), lr = 0.001)

        # train the model on given epochs
        for n in range(epoch):
            # do ReLU optimization
            y_pred_ReLU = ReLU_model(X_train)
            loss_ReLU = loss_fn(y_pred_ReLU, y_train)
            ReLU_optimizer.zero_grad()
            loss_ReLU.backward()
            ReLU_optimizer.step()

            # do ReLU optimization
            y_pred_Sigmoid = Sigmoid_model(X_train)
            loss_Sigmoid = loss_fn(y_pred_Sigmoid, y_train)
            Sigmoid_optimizer.zero_grad()
            loss_Sigmoid.backward()
            Sigmoid_optimizer.step()
            
        # evaluate the models
        ReLU_model.eval()
        Sigmoid_model.eval()

        # get ReLU training accuracy
        y_train_pred_ReLU = ReLU_model(X_train)
        # compute classified prediction using argmax to find highest value
        acc_train_ReLU = (torch.argmax(y_train_pred_ReLU, 1) == torch.argmax(y_train, 1)).float().mean()

        # get training accuracy
        y_test_pred_ReLU = ReLU_model(X_test)
        acc_test_ReLU = (torch.argmax(y_test_pred_ReLU, 1) == torch.argmax(y_test, 1)).float().mean()

        # print current hidden layer size
        print(f'\n\nHidden Layer Size:',layer_size)
        # print current number of epochs
        print(f'Number of Epochs:',epoch)

        # print ReLU accuracies
        print(f'ReLU Model Training Accuracy:', round(acc_train_ReLU.item() * 100, 2), '%')
        print(f'ReLU Model Testing Accuracy:', round(acc_test_ReLU.item() * 100, 2), '%')

        # get ReLU training accuracy
        y_train_pred_Sigmoid = Sigmoid_model(X_train)
        acc_train_Sigmoid = (torch.argmax(y_train_pred_Sigmoid, 1) == torch.argmax(y_train, 1)).float().mean()

        # get training accuracy
        y_test_pred_Sigmoid = Sigmoid_model(X_test)
        acc_test_Sigmoid = (torch.argmax(y_test_pred_Sigmoid, 1) == torch.argmax(y_test, 1)).float().mean()

        # print ReLU accuracies
        print(f'Sigmoid Model Training Accuracy:', round(acc_train_Sigmoid.item() * 100, 2), '%')
        print(f'Sigmoid Model Testing Accuracy:', round(acc_test_Sigmoid.item() * 100, 2), '%')