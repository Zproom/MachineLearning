# linear_regression.py
# This file contains a class that implements simple linear regression.

import numpy as np
import pandas as pd
from training_helper_functions import *

class linear_regression:
    def __init__(self, training_data, target_col):
        """This method initializes a linear_regression instance.

        Args:
            training_data (pd.DataFrame): The training data for the model.
            target_col (list): The name(s) of the target column.
        """
        self.training_data = training_data
        self.target_col = target_col
        self.weights = None

    def train(self, epochs = 1000, learning_rate = 0.01):
        """This method trains a linear regression model using gradient descent.

        Args:
            epochs (int, optional): The number of training epochs to use. Defaults to 1000.
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
        """
        # The feature data is the training data, with the target column removed and an extra column
        # of all ones for the bias.
        feature_data = self.training_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        n_features = feature_data.shape[1]
        # There should be one weight per feature plus an extra weight for the bias term.
        self.weights = np.random.uniform(-0.01, 0.01, n_features)
        actual_y = self.training_data[self.target_col[0]]
        for epoch in range(epochs):
            # Calculate probabilities.
            y_hat = np.dot(feature_data, self.weights)
            # Calculate gradients.
            errors = actual_y - y_hat
            delta_weights = 1/len(feature_data) * np.dot(feature_data.T, errors)
            # Update weights.
            self.weights += learning_rate * delta_weights

    def predict(self, test_data):
        """Generates predictions using the trained linear_regression model.

        Args:
            test_data (pd.DataFrame): The data to make predictions for.

        Returns:
            (pd.Series): The predictions from the model.
        """
        # The feature data is the test data, with the target column removed and an extra column of
        # all ones for the bias.
        feature_data = test_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        return pd.Series(np.dot(feature_data, self.weights))