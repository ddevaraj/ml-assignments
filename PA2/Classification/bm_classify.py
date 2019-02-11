"""
Collaborators: Sushanti Prabhu, Nitika Tanwani
"""

import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5,
                 max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    w = np.insert(w, 0, b)
    ones = np.ones((N, 1))
    X = np.hstack((ones, X))

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        y_changed = np.array(y)
        y_changed[y_changed == 0] = -1
        for i in range(max_iterations):
            y_pred = np.dot(X, w)
            y_pred[y_pred > 0] = 1
            y_pred[y_pred < 0] = -1

            error = y_changed - y_pred
            error[error == -2] = -1
            error[error == 2] = 1
            gradient = np.dot(X.transpose(), error) / len(y_changed)
            w += (step_size * gradient)

            ############################################


    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        for i in range(max_iterations):
            z = np.dot(X, w)
            sig = sigmoid(z)
            gradient = np.dot(np.transpose(X), (sig - y)) / len(y)
            w -= (step_size * gradient)
            ############################################


    else:
        raise "Loss Function is undefined."

    b = w[0]
    w = w[1:]
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1 / (1 + np.exp(-z))
    ############################################

    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic

    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape

    w = np.insert(w, 0, b)
    ones = np.ones((N, 1))
    X = np.hstack((ones, X))

    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        preds = np.where(np.dot(X, w) > 0, 1, 0)
        ############################################

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds
        preds = np.where(sigmoid(np.dot(X, w)) > 0.5, 1, 0)
        ############################################


    else:
        raise "Loss Function is undefined."

    assert preds.shape == (N,)
    return preds


def softmax(x):
    x -= np.max(x)
    sm = (np.exp(x).T / np.sum(np.exp(x), axis=1)).T
    return sm


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0
    w = np.zeros((C, D))
    b = np.zeros(C)

    y_train = np.eye(C)[y]
    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.insert(w, D, b, axis=1)
        X = np.insert(X, D, 1, axis=1)

        for i in range(max_iterations):
            index = np.random.choice(len(X))
            xi = X[index]
            xi = xi.reshape(1, xi.shape[0])
            yi = y_train[index]
            val = softmax(np.dot(xi, np.transpose(w)))
            gradient = np.dot((val - yi).transpose(), xi)
            w -= (step_size * gradient)

        b = w[:, D]
        w = w[:, 0:D]
        ############################################


    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.insert(w, 0, b, axis=1)
        X = np.insert(X, 0, 1, axis=1)

        for i in range(max_iterations):
            val = softmax(np.dot(X, w.transpose()))
            gradient = np.dot((val - y_train).transpose(), X) / len(y_train)
            w -= (step_size * gradient)

        b = w[:, 0]
        w = w[:, 1:]
        ############################################


    else:
        raise "Type of Gradient Descent is undefined."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D
    - b: bias terms of the trained multinomial classifier, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    preds = np.zeros(N)

    X = np.insert(X, 0, 1, axis=1)
    w = np.insert(w, 0, b, axis=1)

    for i in range(X.shape[0]):
        preds[i] = np.argmax(np.dot(X[i], w.transpose()))
    ############################################

    assert preds.shape == (N,)
    return preds