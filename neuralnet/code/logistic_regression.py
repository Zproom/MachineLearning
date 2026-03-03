# logistic_regression.py
# This file contains a class that implements simple logistic regression.

import numpy as np
import pandas as pd
from training_helper_functions import *

class logistic_regression:
    def __init__(self, training_data, target_col):
        """This method initializes a logistic_regression instance.

        Args:
            training_data (pd.DataFrame): The training data for the model.
            target_col (list): The name(s) of the target column.
        """
        self.training_data = training_data
        self.target_col = target_col
        self.weights = None
        if len(target_col) > 1:
            self.type = "multi-class"
        else:
            self.type = "two-class"

    def train(self, epochs = 1000, learning_rate = 0.01):
        """This method trains a logistic regression model using gradient descent.

        Args:
            epochs (int, optional): The number of training epochs to use. Defaults to 1000.
            learning_rate (float, optional): The learning rate. Defaults to 0.01.
        """
        # The feature data is the training data, with the target column removed and an extra column
        # of all ones for the bias.
        feature_data = self.training_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        # Multi-class logistic regression
        if self.type == "multi-class": 
            # This should be a matrix where each target column gets a column, and each feature 
            # (plus the bias) gets a row.
            self.weights = np.random.uniform(-0.01, 0.01, (feature_data.shape[1], 
                                                           len(self.target_col)))
            # Combine the target columns into a numpy ndarray.
            actual_y = self.training_data[self.target_col].to_numpy()
            for epoch in range(epochs):
                # Calculate probabilities.
                y_hats = softmax(np.dot(feature_data, self.weights))
                # Calculate gradients.
                errors = actual_y - y_hats
                delta_weights = np.dot(feature_data.T, errors)
                # Update weights.
                self.weights += learning_rate * delta_weights
        # Two-class logistic regression        
        else:
            # There should be one weight per feature plus an extra weight for the bias term.
            self.weights = np.random.uniform(-0.01, 0.01, feature_data.shape[1])
            actual_y = self.training_data[self.target_col[0]]
            for epoch in range(epochs):
                # Calculate probabilities.
                y_hat = sigmoid(np.dot(feature_data, self.weights))
                # Calculate gradients.
                errors = actual_y - y_hat
                delta_weights = np.dot(feature_data.T, errors)
                # Update weights.
                self.weights += learning_rate * delta_weights

    def predict(self, test_data, threshold = 0.5):
        """Generates predictions using the trained logistic_regression model.

        Args:
            test_data (pd.DataFrame): The data to make predictions for.
            threshold (float): The probability threshold for determining the class for two-class 
            problems.

        Returns:
            (pd.Series): The predictions from the model.
        """
        # The feature data is the test data, with the target column removed and an extra column of
        # all ones for the bias.
        feature_data = test_data.drop(columns = self.target_col)
        feature_data["bias"] = 1
        if self.type == "multi-class":
            y_hats = softmax(np.dot(feature_data, self.weights))
            predicted_classes = (y_hats == y_hats.max(axis = 1, keepdims = True)).astype(int)
            return pd.Series(predicted_classes.tolist()).reset_index(drop = True)
        else:
            y_hat = sigmoid(np.dot(feature_data, self.weights))
            return pd.Series((y_hat >= threshold).astype(int))