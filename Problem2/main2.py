# Homework 1.

'''Problem 2:
Let's buid up a learning algorithm where the NN has a hidden layer to do classification for a cat/non-cat dataset.
It includes the following steps;
    -Initialize the parameters for a two-layer network (a hidden layer and an output layer).
    -Implement the forward propagation module.
        giving you the ACTIVATION function (tanh/relu/sigmoid), you combine a new [LINEAR->tanh->LINEAR->SIGMOID] forward function.
        (the [LINEAR->tanh] forward function at the hidden layer and add a [LINEAR->SIGMOID] at the end (for the final layer ).)
    -Compute the loss.
    -Implement the backward propagation module .
        [LINEAR->SIGMOID] backward and then [LINEAR->RELU] backward
Finally update the parameters.

main2.py--main function to implement the classification by calling all functions in functions2.py
functions2.py--we define all functions needed to implement the classification

Dataset illustration:
You are given a dataset ("data.h5") containing:
- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
Thus, each image is square (height = num_px) and (width = num_px).

Tasks:
    a.Please complete the codes between ### Start code here ### and ### End code here ### [C1 TO C10]
    b.Please answer the questions in the comments [T1]

'''

import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from functions2 import load_dataset, initialize_parameters, forward_propagation, compute_cost,backward_propagation,update_parameters

# Loading the data (cat/non-cat) through the function "load_dataset()" in the file "functions.py"
train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()
print(classes)

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

''' C1 (6') '''
### START CODE HERE ###
n_x = train_x.shape[0] # n_x is the number of neurons at input layer; Requirement: calculate n_x with existing variables;
n_h = 8  # the number of neurons at hidden layer
n_y = 1 # a constant here; the number of neurons at output layer
layers_dims = (n_x, n_h, n_y)
### END CODE HERE ###

##GO to function 'initialize_parameters()' at functions2.py
'''C4 (2')'''
# Initialize parameters dictionary by calling one of the functions in functions2.py
### START CODE HERE ### (1 line of code)
parameters = initialize_parameters(layers_dims)
### END CODE HERE ###

# Loop (gradient descent)
num_iterations = 3000
learning_rate = 0.1
costs = []
print_cost = True
for i in range(0, num_iterations):

    # Forward propagation: LINEAR -> Tanh -> LINEAR -> SIGMOID.
    #GO to function 'forward_propagation()' at file functions2.py
    A2, caches = forward_propagation(train_x, parameters)

    # Compute cost.
    # GO to function 'compute_cost()' at file functions2.py
    cost = compute_cost(A2, train_y)

    # Backward propagation.
    # GO to function 'backward_propagation()' at file functions2.py
    grads = backward_propagation(parameters, caches, train_x, train_y)

    # Update parameters.
    parameters = update_parameters(parameters, grads, learning_rate)

    # Print the cost every 100 training example
    if print_cost and i % 100 == 0:
        print("Cost after iteration %i: %f" % (i, cost))
    if print_cost and i % 100 == 0:
        costs.append(cost)

# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()

''' T1 (15')
Running your algorithm while learning_rate = [0.005, 0.05, 0.1]
Plot the corresponding cost figures for three different learning rate (Y axis represents Cost and X axis represents the number of iterations)
Submit figures as PDF

'''