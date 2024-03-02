# Homework 1.

'''Problem 1:
Let's buid up a learning algorithm(single neuron) with a cat/non-cat dataset.
It includes the following steps;
    1. Load training data and test data
    2. Initializing parameters
    3. Design a function to calculate the prediction value by forward propagation and costs
    4. Design a function to calculate the gradients by backward propagation
    5. Using an optimization algorithm (gradient descent) to minimize the cost
After trained a model, we test the model performance with test dataset;
It includes one step: predict;

Dataset illustration:
You are given a dataset ("data.h5") containing:
- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).
Thus, each image is square (height = num_px) and (width = num_px).

Tasks:
    a.Please complete the codes between ### Start code here ### and ### End code here ### [C1 TO C8]
    b.Please answer the questions in the comments [T1 and T2]

'''

import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from functions1 import load_dataset, initialize_w_b, optimize, predict

# Loading the data (cat/non-cat) through the function "load_dataset()" in the file "functions.py"
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
print(classes)

# View two images in the database; You can comment the following after viewing the images
index1 = 25
index2 = 36 #change number to see different images
plt.imshow(train_set_x_orig[index1])
plt.show()
plt.imshow(train_set_x_orig[index2])
plt.show()

'''C1 (8')'''
### Start code here ###
"Please complete the following four printing sentences to show the corresponding sizes of our data"
"hint: using np.shape() to get sizes of our train and test data"
print("the shape of train_set_x_orig: " + np.shape(train_set_x_orig))
print("the shape of train_set_y: " + np.shape(train_set_y))
print("the shape of test_set_x_orig: " + np.shape(test_set_x_orig))
print("the shape of test_set_y: " + np.shape(test_set_y))
### End code here ###

'''T1 (3')'''
### Answer the following questions: ###

### 1. How many images are there in the training dataset? (m_train)
### Your answer:
### 2. What is the size of those images? (n_x)
### Your answer:
### 3. How many images are there in the test dataset? (m_test)
### Your answer:

### End questions ###


# Reshape the training and test examples
# To implement binary classification, we first flattened each image into a vector
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

'''C2 (8')'''
"Please complete the following four printing sentences to show the corresponding sizes of our data after flattening"
### Start code here ###
print("train_set_x_flatten shape: " + np.shape(train_set_x_flatten))
print("train_set_y shape: " + np.shape(train_set_y))
print("test_set_x_flatten shape: " + np.shape(test_set_x_flatten))
print("test_set_y shape: " + np.shape(train_set_y))
### End code here ###

# Standardize our dataset
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

## Above we prepared our dataset, such as train_set_x, train_set_y, test_set_x, test_set_y(label data);
## After preparing our training data and test data, we need to initialize our parameters;

dim = train_set_x.shape[0]  #obtain the dimention of NN's input layer(dimension of features. In our class, we note this as nx)
w, b = initialize_w_b(dim) #GO to the function "initialize_w_b()" in the file "functions.py"

# print("The initial w and b")
# print ("w = " + str(w))
# print ("b = " + str(b))

## after parameters initialization w and b, we will do forward and backward propagation
'''
    You basically need to write down three steps:
        1) Calculate the prediction of Y and cost by forward propagation
        2) Calculate the gradient for the current parameters by backward propagation.
        3) Update the parameters iteratively using gradient descent rule for w and b.
    
    Therefore, we creat three functions in the file "functions.py"
        1) forwardpropagation()
        2) backpropagation ()
        3) optimize()
    Tips:
            - Write your code step by step for the propagation
'''

num_iterations = 2000
learning_rate=0.005
#GO to the function "optimize()" in the file "functions.py"
params, grads, costs = optimize(w, b, train_set_x, train_set_y,  num_iterations, learning_rate, print_cost = True)
# So far, the parameters of our trained model is represented by w and b in variable 'params'

# Retrieve parameters w and b from dictionary "parameters"
w = params["w"]
b = params["b"]

## Let's visualize the cost; Plot learning curve (with costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate = 0.005")
plt.show()

#After training, let's test/predict pictures by the function 'predict()' in the file 'function.py'
# Predict test set examples
#GO to the function "predict()" in the file "functions.py"
Y_prediction_test = predict(w, b, test_set_x)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

# Example of a picture that was wrongly classified.
index = 5
plt.imshow(test_set_x[:,index].reshape((64, 64, 3)))
plt.show()
if Y_prediction_test[0, index]:
    label = "cat"
else:
    label = "non-cat"
print ("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + label +  "\" picture.")


'''T2 (10')'''
### Answer the following questions: ###

### 1. How many for-loops are there in this code (the code for Problem 1)? Why?
### Your answer:

### End questions ###

