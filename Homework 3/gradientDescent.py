#gradientDescent.py

from computeCost import computeCost 
import random

#functions

# helper function partialSum
# returs the summation portion from partial derivites during gradient descent
def partialSum(x, y, theta, feature):
    partialSum = 0
    tempSum = 0

    #loop through rows in x
    for i in range(len(x)):

        #loop through features in x
        for j in range(len(theta)):
            #calculate h(x)
            tempSum += theta[j] * x[i][j]

        #subtract y
        tempSum -= y[i]

        #multiply by the feature itself
        partialSum += tempSum * x[i][feature]

        #reset the temp sum by itself
        tempSum = 0

    #return the partial sum
    return partialSum


#main function gradientDescent
#returns: theta, cost
def gradientDescent(x_train, y_train, theta, alpha, iters):
    # get data size
    featureSize = x_train.shape

    # iterate through gradient descent algorithm
    for i in range(iters):
        # make a temporary array to hold theta values before updating
        tempTheta = [0] * len(theta)

        # loop through each feature
        for j in range(len(theta)):
            tempTheta[j] = (alpha/len(y_train)) * partialSum(x_train, y_train, theta, j)
            
        # loop through theta again to simultaneously update
        for k in range(len(theta)):
            theta[k] -= tempTheta[k]

    return theta