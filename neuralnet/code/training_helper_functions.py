# training_helper_functions.py
# This file contains helper functions for training the machine learning models.

import numpy as np

def sigmoid(x):
    """This is an implementation of the sigmoid function.

    Args:
        x (float): The input to the sigmoid function.

    Returns:
        (float): The result of applying the sigmoid function to the input.
    """
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    """This is an implementation of the sigmoid derivative function.

    Args:
        x (float): The input to the sigmoid derivative function.

    Returns:
        (float): The result of applying the sigmoid derivative function to the input.
    """
    return sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
    """This is an implementation of the softmax function.

    Args:
        x (np.ndarray): This contains the dot products of the feature data and the weights for each 
        class.

    Returns:
        (np.ndarray): This contains the softmax probability distribution for the target columns.
    """
    exp_x = np.exp(x)
    return exp_x/exp_x.sum(axis = 1, keepdims = True)