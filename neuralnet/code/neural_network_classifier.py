# neural_network_classifier.py
# This file contains a class that implements a neural network for classification.

import numpy as np
import pandas as pd
from training_helper_functions import *

class neural_network_classifier:
    def __init__(self, training_data, target_col, hidden1_size, hidden2_size):
        """This method initializes a neural_network_classifier instance.

        Args:
            training_data (pd.DataFrame): The training data for the model.
            target_col (list): The name(s) of the target column.
            hidden1_size (int): The number of nodes in the first hidden layer.
            hidden2_size (int): The number of nodes in the second hidden layer.
        """
        self.training_data = training_data
        self.target_col = target_col
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.weights1 = None
        self.weights2 = None
        self.weights3 = None
        if len(target_col) > 1:
            self.type = "multi-class"
        else:
            self.type = "two-class"

    def train(self, epochs = 1000, learning_rate = 0.01):
        """This method trains a neural network classifier using gradient descent.

        Args:
            epochs (int, optional): The number of training epochs to use. Defaults to 1000.
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
        """
        # The feature data is the training data, with the target column removed and an extra column
        # of all ones for the bias.
        feature_data = self.training_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        # Initialize weights for all three layers.
        # Weights on the edges connecting the input data to the first hidden layer.
        self.weights1 = np.random.uniform(-0.01, 0.01, (feature_data.shape[1], self.hidden1_size))
        # Weights on the edges connecting the first hidden layer to the second hidden layer. 
        self.weights2 = np.random.uniform(-0.01, 0.01, (self.hidden1_size, self.hidden2_size))
        if self.type == "multi-class": 
            # Combine the target columns into a numpy ndarray.
            actual_y = self.training_data[self.target_col].to_numpy()
            # Weights on the edges connecting the second hidden layer to the output layer. 
            self.weights3 = np.random.uniform(-0.01, 0.01, (self.hidden2_size, 
                                                            len(self.target_col)))
        else:
            actual_y = self.training_data[self.target_col[0]]
            self.weights3 = np.random.uniform(-0.01, 0.01, self.hidden2_size)
        for epoch in range(epochs):
            # Forward pass
            o1 = np.dot(feature_data, self.weights1)
            y_hats1 = sigmoid(o1)
            o2 = np.dot(y_hats1, self.weights2)
            y_hats2 = sigmoid(o2)
            o3 = np.dot(y_hats2, self.weights3)
            if self.type == "multi-class":
                y_hats3 = softmax(o3)
            else:
                y_hats3 = sigmoid(o3)
            # Backpropagation
            do3 = y_hats3 - actual_y # Assume loss is cross-entropy loss.
            dweights3 = np.dot(y_hats2.T, do3)
            if self.type == "multi-class":
                dy2 = np.dot(do3, self.weights3.T)
            else:
                dy2 = np.dot(do3.values.reshape(-1, 1), self.weights3.reshape(1, -1))
            do2 = dy2 * sigmoid_derivative(o2)
            dweights2 = np.dot(y_hats1.T, do2)
            dy1 = np.dot(do2, self.weights2.T)
            do1 = dy1 * sigmoid_derivative(o1)
            dweights1 = np.dot(feature_data.T, do1)
            # Weight updates
            self.weights3 -= learning_rate * dweights3
            self.weights2 -= learning_rate * dweights2
            self.weights1 -= learning_rate * dweights1

    def predict(self, test_data, threshold = 0.5):
        """Generates predictions using the trained neural network model.

        Args:
            test_data (pd.DataFrame): The data to make predictions for.
            threshold (float): The probability threshold for determining the class for two-class 
            problems.

        Returns:
            (pd.Series): The predictions from the model.
        """
        feature_data = test_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        o1 = np.dot(feature_data, self.weights1)
        y_hats1 = sigmoid(o1)
        o2 = np.dot(y_hats1, self.weights2)
        y_hats2 = sigmoid(o2)
        o3 = np.dot(y_hats2, self.weights3)
        if self.type == "multi-class":
            y_hats3 = softmax(o3)
            predicted_classes = (y_hats3 == y_hats3.max(axis = 1, keepdims = True)).astype(int)
            return pd.Series(predicted_classes.tolist()).reset_index(drop = True)
        else:
            y_hats3 = sigmoid(o3)
            return pd.Series((y_hats3 >= threshold).astype(int))