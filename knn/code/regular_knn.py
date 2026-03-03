# regular_knn.py
# This file defines classes to implement regular k-nearest neighbor classification and regression.

import pandas as pd
import numpy as np
import random
from distance_functions import *

class knn_classifier:
    def __init__(self, k, training_data):
        """This method initializes a knn_classifier instance.

        Args:
            k (int): The number of neighbors to check to classify a new data point.
            training_data (pd.DataFrame): The training data for the model.
        """
        self.k = k
        self.training_data = training_data

    def predict(self, test_data, target_col):
        """This method classifies new examples (test data) using the k-nearest neighbors 
        algorithm.

        Args:
            test_data (pd.DataFrame): The new data to make predictions for.
            target_col (list): The name of the target column(s) in test_data. There may be multiple 
            target columns. For example, if the target value takes on more than two unique values 
            (car evaluation dataset).

        Returns:
            (pd.Series): A class prediction for every new example.
        """
        # Create copies of the training and test data with the target column(s) removed. The target
        # column should not be used to calculate Euclidean distances.
        training_data_no_target = self.training_data.drop(columns = target_col)
        test_data_no_target = test_data.drop(columns = target_col)
        predictions = [None] * len(test_data_no_target) # Preallocate a list.
        # Loop through every observation in the test data.
        for test_index, test_row in test_data_no_target.iterrows():
            distances = [None] * len(training_data_no_target) # Preallocate a list.
            # Compute the Euclidean distance between the test observation and every observation in 
            # the training data.
            distances = distance_num(test_row, training_data_no_target)
            # Sort the distances (smallest to largest).
            distances = distances.sort_values()
            # Find the class labels of the k-nearest neighbors.
            k_nearest_distances = distances.iloc[0:self.k]
            k_nearest_indices = list(k_nearest_distances.index)
            k_nearest_neighbors_data = self.training_data.iloc[k_nearest_indices]
            k_nearest_neighbors_classes = k_nearest_neighbors_data[target_col]
            if len(target_col) > 1:
                k_nearest_neighbors_classes = pd.Series(
                    k_nearest_neighbors_classes.values.tolist())    
            # Count the class labels of the k-nearest neighbors.
            k_nearest_counts = k_nearest_neighbors_classes.value_counts()
            # Assign the class that appears the most among the k-nearest neighbors. 
            highest_count = k_nearest_counts.iloc[0]
            if len(target_col) > 1:
                highest_count_class = k_nearest_counts.index.tolist()[0]
            else:
                highest_count_class = \
                k_nearest_counts.index.get_level_values(target_col[0]).tolist()[0]
            # Handle ties if appropriate. Randomly (and uniformly) choose the class from among the 
            # tied classes.
            tied_classes = k_nearest_counts[k_nearest_counts == highest_count]
            if len(tied_classes) > 1:
                if len(target_col) > 1:
                    highest_count_class = random.choice(tied_classes.index.tolist())
                else:
                    highest_count_class = \
                    random.choice(k_nearest_counts.index.get_level_values(target_col[0]).tolist())
            # Store the prediction.
            predictions[test_index] = highest_count_class
        return pd.Series(predictions)

class knn_regressor:
    def __init__(self, k, bandwidth, training_data, cyclical = False):
        """This method initializes a knn_regressor instance.

        Args:
            k (int): The number of neighbors to check to classify a new data point.
            bandwidth (float): The bandwidth of the Gaussian kernel. A larger value means the 
            weight of a point decreases more quickly as the distance increases.
            training_data (pd.DataFrame): The training data for the model.
            cyclical (bool): Does the training data contain cyclical features (only applies to forestfires data).
        """
        self.k = k
        self.bandwidth = bandwidth
        self.training_data = training_data
        self.cyclical = cyclical

    def predict(self, test_data, target_col):
        """This method predicts new examples (test data) using the k-nearest neighbors 
        algorithm (regression).

        Args:
            test_data (pd.DataFrame): The new data to make predictions for.
            target_col (list): The name of the target column(s) in test_data. There may be multiple 
            target columns. For example, if the target value takes on more than two unique values 
            (car evaluation dataset).

        Returns:
            (pd.Series): A class prediction for every new example.
        """
        # Create copies of the training and test data with the target column(s) removed. The target
        # column should not be used to calculate Euclidean distances.
        training_data_no_target = self.training_data.drop(columns = target_col)
        test_data_no_target = test_data.drop(columns = target_col)
        predictions = [None] * len(test_data_no_target) # Preallocate a list.
        # Loop through every observation in the test data.
        for test_index, test_row in test_data_no_target.iterrows():
            distances = [None] * len(training_data_no_target) # Preallocate a list.
            # Compute the Euclidean distance between the test observation and every observation in 
            # the training data.
            distances = distance_num(test_row, training_data_no_target, self.cyclical)
            # Sort the distances (smallest to largest).
            distances = distances.sort_values()
            # Find the response values of the k-nearest neighbors.
            k_nearest_distances = distances.iloc[0:self.k]
            k_nearest_indices = list(k_nearest_distances.index)
            k_nearest_neighbors_data = self.training_data.iloc[k_nearest_indices]
            k_nearest_neighbors_response = k_nearest_neighbors_data[target_col[0]]
            # Compute the average response value of the k-nearest neighbors and store the 
            # prediction. Use a weighted average (Gaussian kernel).
            weights = np.exp(-1*self.bandwidth*k_nearest_distances)
            predictions[test_index] = np.average(k_nearest_neighbors_response, 
                                                 weights = weights)
        return pd.Series(predictions)