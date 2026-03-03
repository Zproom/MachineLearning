# autoencoder.py
# This file contains a class that implements an autoencoder.

import numpy as np
import pandas as pd
from training_helper_functions import *

class autoencoder:
    def __init__(self, training_data, target_col, hidden_size):
        """This method initializes a neural_network_classifier instance.

        Args:
            training_data (pd.DataFrame): The training data for the model.
            target_col (list): The name(s) of the target column.
            hidden_size (int): The number of nodes in the hidden layer.
        """
        self.training_data = training_data
        self.target_col = target_col
        self.hidden_size = hidden_size
        self.weights1 = None
        self.weights2 = None
        if hidden_size >= len(training_data):
            raise Exception("The number of hidden nodes must be less than the number of input " +
                            "nodes.")

    def train(self, epochs = 1000, learning_rate = 0.01):
        """This method trains an autoencoder using gradient descent.

        Args:
            epochs (int, optional): The number of training epochs to use. Defaults to 1000.
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
        """
        feature_data = self.training_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        n_obs = feature_data.shape[0]
        n_features = feature_data.shape[1]
        # Initialize weights.
        # Weights on the edges connecting the input data to the hidden layer.
        self.weights1 = np.random.uniform(-0.01, 0.01, (n_features, self.hidden_size))
        # Weights on the edges connecting the hidden layer to the output layer. 
        self.weights2 = np.random.uniform(-0.01, 0.01, (self.hidden_size, n_features))
        for epoch in range(epochs):
            # Forward pass
            # Encoder
            o1 = np.dot(feature_data, self.weights1)
            y_hats1 = sigmoid(o1)
            # Decoder
            o2 = np.dot(y_hats1, self.weights2)
            y_hats2 = sigmoid(o2) # Final output (reconstructed input data)
            # Backpropagation
            dy2 = 2/n_obs * (y_hats2 - feature_data)
            do2 = dy2 * sigmoid_derivative(o2)
            dweights2 = np.dot(y_hats1.T, do2)
            dy1 = np.dot(do2, self.weights2.T)
            do1 = dy1 * sigmoid_derivative(o1)
            dweights1 = np.dot(feature_data.T, do1)
            # Weight updates
            self.weights2 -= learning_rate * dweights2
            self.weights1 -= learning_rate * dweights1