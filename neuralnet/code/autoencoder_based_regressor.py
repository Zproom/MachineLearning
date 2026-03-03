# autoencoder_based_regressor.py
# This file contains a class that implements autoencoder-based regression.

import numpy as np
import pandas as pd
from training_helper_functions import *
from autoencoder import *

class autoencoder_based_regressor():
    def __init__(self, training_data, target_col, hidden1_size, hidden2_size):
        """This method initializes an autoencoder_based_regressor instance.

        Args:
            training_data (pd.DataFrame): The training data for the model.
            target_col (list): The name(s) of the target column.
            hidden1_size (int): The number of nodes in the first (autoencoder) hidden layer.
            hidden2_size (int): The number of nodes in the second hidden layer.
        """
        self.training_data = training_data
        self.target_col = target_col
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.weights1 = None
        self.weights2 = None
        self.weights3 = None

    def train(self, epochs = 1000, learning_rate = 0.01):
        """This method trains an autoencoder-based classifier using gradient descent.

        Args:
            epochs (int, optional): The number of training epochs to use. Defaults to 1000.
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
        """
        # Pretrain the autoencoder.
        autoencoder_layer = autoencoder(self.training_data, self.target_col, self.hidden1_size)
        autoencoder_layer.train()
        # Copy the first set of weights from the autoencoder. These won't be updated.
        self.weights1 = autoencoder_layer.weights1
        # Add a regression layer on top of the autoencoder's hidden layer. Train this
        # regression layer, holding the weights in the autoencoder's hidden layer fixed.
        feature_data = self.training_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        n_obs = len(feature_data)
        self.weights2 = np.random.uniform(-0.01, 0.01, (self.hidden1_size, self.hidden2_size))
        self.weights3 = np.random.uniform(-0.01, 0.01, self.hidden2_size)
        actual_y = self.training_data[self.target_col[0]]
        for epoch in range(epochs):
            # Forward pass
            o1 = np.dot(feature_data, self.weights1)
            y_hats1 = sigmoid(o1)
            o2 = np.dot(y_hats1, self.weights2)
            y_hats2 = sigmoid(o2)
            o3 = np.dot(y_hats2, self.weights3)
            y_hats3 = o3 # Linear output layer
            # Backpropagation
            # Assume loss is MSE = 1/n * sum(y_pred - actual_y)^2.
            do3 = 2/n_obs * (y_hats3 - actual_y)
            dweights3 = np.dot(y_hats2.T, do3)
            dy2 = np.dot(do3.values.reshape(-1, 1), self.weights3.reshape(1, -1))
            do2 = dy2 * sigmoid_derivative(o2)
            dweights2 = np.dot(y_hats1.T, do2)
            dy1 = np.dot(do2, self.weights2.T)
            do1 = dy1 * sigmoid_derivative(o1)
            dweights1 = np.dot(feature_data.T, do1)
            # Weight updates
            self.weights3 -= learning_rate * dweights3
            self.weights2 -= learning_rate * dweights2

    def predict(self, test_data):
        """Generates predictions using the trained neural network model.

        Args:
            test_data (pd.DataFrame): The data to make predictions for.

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
        y_hats3 = o3
        return pd.Series(y_hats3)