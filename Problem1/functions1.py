#import packages
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def initialize_w_b(dimension):
    ## Now we initialize our parameters w and b
    """
            dimension -- size of the w vector we want (this depends on the number of neuron in the (l-1)-th layer)

            Returns:
            w -- initialized vector of shape (dimension 1)
            b -- initialized scalar (corresponds to the bias)
            """

    '''C3 (2')'''
    "Please write one sentence code to initialize w as a zeros vector"
    ### START CODE HERE ### (1 line of code)
    "hint: initialized zeros vector of shape (dimension, 1) using np.zeros()"
    w = np.zeros(1)
    b = 0
    ### END CODE HERE ###
    print (w.shape == (dimension, 1))
    return w,b

def forwardpropagation(w,b,X,Y):

    """
            Implement the forward propagation and the cost function
            Arguments:
            w -- weights, a numpy array of size (num_px * num_px * 3, 1)
            b -- bias, a scalar
            X -- data of size (num_px * num_px * 3, number of examples)
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

            Return:
                A--Output of neuron
                cost -- negative log-likelihood cost for logistic regression

            Tips:
            - Write your code step by step for the propagation
            """
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    '''C4 (4')'''
    "Complete the following two sentences to implement forward propagation"
    ### START CODE HERE ### (2 lines of code)
    "hint1: z = the transpose of W * X + b; one line of code"
    z = np.dot(w.T, X) + b
    "hint2: A = sigmoid (z); one line of code to compute activation"
    A = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    return A, cost


def backpropagation(X, Y, A):
    # BACKWARD PROPAGATION (TO FIND GRAD)
    """
            Implement the backward propagation and calculate the gradient
            Arguments:
            X -- data of size (num_px * num_px * 3, number of examples)
            Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

            Return:
            dw -- gradient of the loss with respect to w, thus same shape as w
            db -- gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1]
    '''C5 (4')'''
    "Please complete the following two sentences to calculate the derivatives of w and b, respectively"
    ### START CODE HERE ### (2 lines of code)
    " hint: dw is equal to: X times the transpose of (A-Y) and then divided by m, which can be retrieved by the computation graph"
    dw = (X * np.transpose(A-Y)) / m
    "hint: db is equal to: summation of (A-Y) and then divided by m"
    db = np.sum(A-Y) / m
    ### END CODE HERE ###

    grads = {"dw": dw,
             "db": db}

    return grads

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically implement forward propagation, backward propagation, and update parameters iteratively.
    """

    costs = []

    for i in range(num_iterations):

        "We implement forward propagation and back propagation iteratively."
        # GO to the function "forwardpropagation()" above
        # GO to the function "backpropagation()" above

        '''C6 (4')'''
        "Please calculate the forward propagation result A, the cost, and the gradient by calling two of functions in this file"
        ### START CODE HERE (2 lines) ###
        "hint: call two function to implement forward propagation and backward propagation"
        A, cost = forwardpropagation(w, b, X, Y)
        grads = backpropagation(X, Y, A)
        ### END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        '''C7 (4')'''
        "Please update the parameters w, b by gradient descent"
        ### START CODE HERE (2 lines) ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        ### END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, test_set_x):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    test_set_x -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = test_set_x.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(test_set_x.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    '''C8 (4')'''
    "Predict the probability of 'test_set_x' being a cat"
    ### START CODE HERE ### (2 lines of code)
    z = np.dot(w.T, test_set_x) + b
    A = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###

    for i in range(A.shape[1]):
        # Convert probabilities a[0,i] to actual predictions p[0,i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction