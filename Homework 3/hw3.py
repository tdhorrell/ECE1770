# Timothy Horrell
# ECE1770 Homework 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#--------------------------------
#           Problem 1
#--------------------------------

# import data into Pandas dataframe
restaurant_profit_df = pd.read_csv("input/HW3_ex1data1.txt", sep=",", header=None)
restaurant_profit_df.columns = ["Population", "Profit"]

'''
# visualize data with scatterplot
restaurant_profit_df.plot.scatter(x='Population', y='Profit')
plt.title("Population vs. Profit for Restaurant in Cities")
plt.savefig('output/hw3_1b')
'''

# append bias term and reshape
restaurant_profit_data = restaurant_profit_df.to_numpy()
X_data = np.expand_dims((restaurant_profit_data[:,0]), 1)
X_data = np.append(np.ones_like(X_data), X_data, axis=1)
y_data = restaurant_profit_data[:,1]
theta = np.zeros((X_data.shape[1]))

# compute cost function in computeCost.py

# gradient descent function in gradientDescent.py

# run the function for 1000 times
theta = gradientDescent(X_data, y_data, theta, 0.01, 1000)

# print final theta and cost
print(f'Final theta: ',theta)
print(f'Cost of theta:',computeCost(X_data, y_data, theta))

# plot the line over the data
'''
restaurant_profit_df.plot.scatter(x='Population', y='Profit')
xSpace = np.linspace(5, 22.5, 100)
plt.plot(xSpace, theta[0] + theta[1]*xSpace, color='green')
plt.legend(['Profit Data' , 'approximation line'], loc = 'upper left')
plt.title("Population vs. Profit for Restaurant in Cities")
plt.savefig('output/hw3_1f')
'''

# create function for predicting profit of the city
def predictProfit(population, theta):
    return theta[0] + theta[1]*population

# print prediction for given cases
print(f'\nProfit of city with 3,500 residents:',predictProfit(3.5, theta))
print(f'Profit of city with 70,000 residents:',predictProfit(70, theta))


#--------------------------------
#           Problem 2
#--------------------------------

# Read and describe data
iris_df = pd.read_csv("input/HW3_iris.csv", sep=",")
print(f'Iris data shape:',iris_df.shape)
print(f'Iris data type:',type(iris_df))
print(f'\n',iris_df.head(3))
print(f'\n',iris_df.describe())

# Get data into numpy arrays in the correct form
iris_data = iris_df.to_numpy()
iris_X_data = iris_data[:, 1:5]
iris_y_data = iris_data[:,5]

# do label encoding for y
le = LabelEncoder()
le.fit(iris_y_data)
iris_y_data = le.transform(iris_y_data)

# seperate the data using sklearn
X_train, X_test, y_train, y_test = train_test_split(iris_X_data, iris_y_data, train_size=0.7, random_state=7)

# train the KNN classifier
neigh = knc(n_neighbors=5).fit(X_train, y_train)
print(f'\n5NN Output:',le.inverse_transform(neigh.predict(X_test)))

# train the logistic regression
logreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=10000).fit(X_train, y_train)
print(f'\nLogistic Regression Accuracy:',logreg.score(X_test,y_test))