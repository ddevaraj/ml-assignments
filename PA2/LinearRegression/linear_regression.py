"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = np.square(np.subtract(np.dot(X,w),y)).mean()
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    #	TODO 2: Fill in your code here #
    #####################################################
    w = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y))
    return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    xTx = np.dot(X.transpose(),X)
    while np.absolute(np.linalg.eigvals(xTx)).min() < pow(10, -5):
        lambdaI = np.multiply(pow(10,-1),np.identity(xTx.shape[0]))
        xTx = np.add(xTx, lambdaI)
    w = np.dot(np.linalg.inv(xTx), np.dot(X.transpose(), y))
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    w = np.dot(np.linalg.inv(np.add(np.dot(X.transpose(), X),np.multiply(lambd,np.identity(X.shape[1])))), np.dot(X.transpose(),y))
    return w


###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    min_mse = float('inf')
    lambds = [-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    bestlambda = 0
    for power in lambds:
        lambd = pow(10,power)
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        mse = mean_square_error(w, Xval, yval)
        if mse < min_mse:
            min_mse = mse
            bestlambda = lambd
        # lambds *= 10
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################
    Y = np.array(X)

    for i in range(2,power+1):
        # print('in for')
        # print(X.shape)
        X = np.append(X, np.power(Y,i), axis=1)
    return X


