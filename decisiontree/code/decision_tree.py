# decision_tree.py
# This file defines a class to implement decision trees with the ID3 algorithm.

import numpy as np
import pandas as pd
from node import *
from training_helper_functions import *

class decision_tree:
    def __init__(self, training_data, target_col, type, output_file = None):
        """This method initializes a decision_tree instance.

        Args:
            training_data (pd.DataFrame): The training data for the model.
            target_col (str): The name of the target column.
            type (str): The type of decision tree. It can either be "classification" or 
            "regression".
            output_file (str): The file to write output to for analysis. This stores data on the 
            number of nodes vs. the error during the pruning process. This is only needed for
            pruning.
        """
        if type != "classification" and type != "regression":
            raise Exception("type must be 'classification' or 'regression'.")
        
        self.training_data = training_data
        self.target_col = target_col
        self.type = type
        self.output_file = output_file
        self.root = None # Root node of the tree
        self.is_trained = False # Has the tree been trained?
        self.is_pruned = False # Has the tree been pruned?

    def count_nodes(self, start_node):
        """This method counts the number of nodes in the decision tree.

        Args:
            start_node (node): The root of the tree.

        Returns:
            (int): The number of nodes in the tree.
        """
        # We're at a leaf node.
        if start_node.children is None:
            return 0
        total = 1
        for child in start_node.children.values():
            total += self.count_nodes(child)
        return total
    
    def print_tree(self, start_node, depth):
        """This method recursively prints all the nodes in a decision tree.

        Args:
            start_node (node): The root of the tree.
            depth (int): The current depth of the tree (depth of the root node is 0).
        """
        indent = "  " * depth
        for child in start_node.children.values():
            if child.feature is not None:
                if "_num" in child.feature:
                    print(indent + "Decision Node: " + child.feature + ", Split Value = " + \
                          str(child.numeric_split))
                else:
                    print(indent + "Decision Node: " + child.feature)
                self.print_tree(child, depth + 1)
            else:
                print(indent + "Leaf Node, Prediction = " + str(child.prediction))

    def calculate_leaf_value(self, input_data):
        """This method calculates the predicted value at a leaf node during training.

        Args:
            input_data (pd.DataFrame): The data in the tree partition.

        Returns:
            (str): The predicted value for a leaf node.
        """
        if self.type == "classification":
            # The prediction is the most frequent class in the target column for the remaining data. If
            # there are multiple modes, use the first one that is returned.
            most_frequent_target_vals = input_data[self.target_col].mode().tolist()[0] 
            return str(most_frequent_target_vals)
        else: # Regression
            # The prediction is the average target value in the target column for the remaining data.
            return input_data[self.target_col].mean()

    def find_best_split_classification(self, input_data, remaining_features):
        """This method finds the best feature to split on for classification trees.

        Args:
            input_data (pd.DataFrame): The data in the tree partition.
            remaining_features (list): A list of strings containing the names of the features that
            have not been split on yet. This implementation of decision trees only uses a feature
            once for splitting.

        Returns:
            (tuple) A two-element tuple containing:
                (str): The name of the best feature to split on.
                (float): If the feature is numeric, this is the value to split the data in half on.
        """
        # Calculate the entropy of the data.
        entropy = calculate_entropy(input_data, self.target_col)
        best_gain_ratio = -np.inf # Initialize to an extremely low value.
        best_feature = None
        best_split_value = None
        for feature in remaining_features:
            # The feature is numeric. We need to find a value where we can split the data into two 
            # subsets.
            if "_num" in feature: 
                # Only consider splits at midpoints between feature values where the target_col 
                # changes.
                input_data_sorted = input_data.sort_values(by = feature).reset_index(drop = True)
                midpoints = []
                for i in range(len(input_data_sorted) - 1):
                    # Class change
                    if input_data_sorted[self.target_col].iloc[i] != \
                    input_data_sorted[self.target_col].iloc[i + 1]:
                        midpoint = (input_data_sorted[feature].iloc[i] + \
                                    input_data_sorted[feature].iloc[i + 1])/2
                        midpoints.append(midpoint)
                midpoints = np.unique(midpoints)
                best_gain_ratio_all_splits = -np.inf
                best_split_value_all_splits = None
                # Calculate the gain ratio for all possible splits.
                for m in midpoints:
                    gain_ratio = calculate_gain_ratio_num(input_data, self.target_col, feature, m, 
                                                          entropy)
                    if gain_ratio > best_gain_ratio_all_splits:
                        best_gain_ratio_all_splits = gain_ratio
                        best_split_value_all_splits = m
                # Update best feature and split if appropriate.
                if best_gain_ratio_all_splits > best_gain_ratio:
                    best_gain_ratio = best_gain_ratio_all_splits
                    best_feature = feature
                    best_split_value = best_split_value_all_splits
            # The feature is categorical. Split using all unique values of the feature.
            else:
                gain_ratio = calculate_gain_ratio_cat(input_data, self.target_col, feature, 
                                                      entropy)
                # Update best feature if appropriate.
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature
                    # Set to None in case the previous best feature is numeric.
                    best_split_value = None 
        return best_feature, best_split_value

    def find_best_split_regression(self, input_data, remaining_features):
        """This method finds the best feature to split on for regression trees.

        Args:
            input_data (pd.DataFrame): The data in the tree partition.
            remaining_features (list): A list of strings containing the names of the features that
            have not been split on yet. This implementation of decision trees only uses a feature
            once for splitting.

        Returns:
            (tuple) A two-element tuple containing:
                (str): The name of the best feature to split on.
                (float): If the feature is numeric, this is the value to split the data in half on.
        """
        best_mse = np.inf # Initialize to an extremely high value.
        best_feature = None
        best_split_value = None
        for feature in remaining_features:
            # The feature is numeric. We need to find a value where we can split the data into two 
            # subsets.
            if "_num" in feature: 
                # Only consider splits at midpoints between feature values where the target_col 
                # changes.
                input_data_sorted = input_data.sort_values(by = feature).reset_index(drop = True)
                midpoints = []
                for i in range(len(input_data_sorted) - 1):
                    # Class change
                    if input_data_sorted[self.target_col].iloc[i] != \
                    input_data_sorted[self.target_col].iloc[i + 1]:
                        midpoint = (input_data_sorted[feature].iloc[i] + \
                                    input_data_sorted[feature].iloc[i + 1])/2
                        midpoints.append(midpoint)
                midpoints = np.unique(midpoints)
                best_mse_all_splits = np.inf
                best_split_value_all_splits = None
                # Calculate the mean squared error (MSE) for all possible splits.
                for m in midpoints:
                    mse = calculate_mse_num(input_data, self.target_col, feature, m)
                    if mse < best_mse_all_splits:
                        best_mse_all_splits = mse
                        best_split_value_all_splits = m
                # Update best feature and split if appropriate.
                if best_mse_all_splits < best_mse:
                    best_mse = best_mse_all_splits
                    best_feature = feature
                    best_split_value = best_split_value_all_splits
            # The feature is categorical. Split using all unique values of the feature.
            else:
                mse = calculate_mse_cat(input_data, self.target_col, feature)
                # Update best feature if appropriate.
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature
                    # Set to None in case the previous best feature is numeric.
                    best_split_value = None 
        return best_feature, best_split_value

    def train(self):
        """This method trains the classification tree using the training data and using the gain
        ratio as the splitting criterion. It uses the ID3 algorithm.
        """
        remaining_features = [col for col in self.training_data.columns if col != self.target_col]
        self.root = self.build_tree(self.training_data, remaining_features)
        self.is_trained = True
    
    def build_tree(self, input_data, remaining_features):
        """This method is a helper for train() and builds the decision tree recursively.

        Args:
            input_data (pd.DataFrame): The data in the tree partition.
            remaining_features (list): A list of strings containing the names of the features that
            have not been split on yet. This implementation of decision trees only uses a feature
            once for splitting.
        """
        # Check if stopping conditions have been met.
        # All target values are the same or all features have already been split on.
        if len(np.unique(input_data[self.target_col]).tolist()) == 1 or \
            len(remaining_features) == 0:
            prediction = self.calculate_leaf_value(input_data)
            return node(prediction = prediction)
        # If the stopping conditions have not been met, continue building the tree.
        # Find the best split.
        if self.type == "classification":
            split_feature, split_value = self.find_best_split_classification(input_data, 
                                                                             remaining_features)
        else: # Regression
            split_feature, split_value = self.find_best_split_regression(input_data, 
                                                                             remaining_features)
        # Remove the best feature from the list of remaining features.
        remaining_features.remove(split_feature)
        # Split the data based on the best feature.
        # If the best feature is numeric, split the data into two subsets.
        if split_value is not None:
            subset_below, subset_above = split_num(input_data, split_feature, split_value)
            # Check if any of the new subsets of the data are empty. If a subset is empty, create a
            # leaf node where the prediction is the most common class from the data in the current
            # partition.
            average_target_value = self.calculate_leaf_value(input_data)
            if len(subset_below) == 0:
                left_subtree = node(prediction = average_target_value)
            else:
                left_subtree = self.build_tree(subset_below, remaining_features)
            if len(subset_above) == 0:
                right_subtree = node(prediction = average_target_value)
            else:
                right_subtree = self.build_tree(subset_above, remaining_features)
            return node(feature = split_feature, numeric_split = split_value, 
                        children = dict(below = left_subtree, above = right_subtree),
                        pruned_prediction = average_target_value)
        # If the best feature is categorical, split the data into one subset per unique feature 
        # value.
        else:
            subsets, feature_values = split_cat(input_data, split_feature)
            children = dict()
            most_common_target_class = self.calculate_leaf_value(input_data)
            for i in range(len(subsets)):
                # Check if the subset is empty. If it's empty, create a leaf node where the 
                # prediction is the most common class from the data in the current partition.
                if len(subsets[i]) == 0:
                    subtree = node(prediction = most_common_target_class)
                subtree = self.build_tree(subsets[i], remaining_features)
                children[str(feature_values[i])] = subtree
            # Create a special "unknown" node so that if a new feature value is encountered during
            # testing, there is still a way to make a prediction. Use the most common class from 
            # the data in the current partition.
            children["unknown"] = node(prediction = most_common_target_class)
            return node(feature = split_feature, children = children, 
                        pruned_prediction = most_common_target_class)
        
    def predict(self, test_data):
        """This method generates predictions for a test set using the trained decision tree.

        Args:
            test_data (pd.DataFrame): Test data to make predictions for.

        Returns:
            (pd.Series): Predictions, one per test observation.
        """
        if not self.is_trained:
            raise Exception("The decision tree must be trained before making predictions.")
        predictions = []
        for test_index, test_row in test_data.iterrows():
            predictions.append(self.find_prediction(test_row, self.root))
        return pd.Series(predictions)
    
    def find_prediction(self, test_obs, start_node):
        """This method traverses the decision tree to find a prediction for a single observation.

        Args:
            test_obs (pd.Series): A test observation.
            start_node (node): The node in the tree to start the search at.

        Returns:
            (str): A prediction.
        """
        # Stopping condition: We're at a leaf node.
        if start_node.prediction is not None:
            return start_node.prediction
        # Continue traversing the tree.
        # Check which feature is being used to make a split.
        split_feature = start_node.feature
        test_obs_feature_val = test_obs[split_feature]
        # If the split feature is numeric, check whether the observation's corresponding feature
        # value falls below or above the node's numeric_split attribute and traverse that part of 
        # the tree.
        if "_num" in split_feature:
            if test_obs_feature_val <= start_node.numeric_split:
                return self.find_prediction(test_obs, start_node.children["below"])
            else:
                return self.find_prediction(test_obs, start_node.children["above"])
        # If the split feature is categorical, find which feature value the observation has and 
        # check if a corresponding child node exists. If a child node doesn't exist, use the
        # "unknown" child node.
        else:
            test_obs_feature_val = str(test_obs_feature_val)
            if test_obs_feature_val not in list(start_node.children.keys()):
                return self.find_prediction(test_obs, start_node.children["unknown"])
            else:
                return self.find_prediction(test_obs, start_node.children[test_obs_feature_val])
            
    def prune(self, validation_data):
        """This method implements reduced-error pruning. It prunes the trained decision tree.

        Args:
            validation_data (pd.DataFrame): Validation data for measuring performance.
        """
        if not self.is_trained:
            raise Exception("The decision tree must be trained before pruning.")
        self.prune_tree(self.root, validation_data)
        self.is_pruned = True
    
    def prune_tree(self, start_node, validation_data):
        """This method is a helper for prune() and prunes the classification tree recursively.

        Args:
            start_node (node): The node in the tree to start the pruning at.
            validation_data (pd.DataFrame): Validation data for measuring performance.
        """
        # Stopping condition: We're at a leaf node.
        if start_node.prediction is not None:
            return
        # We're at a non-leaf (decision) node.
        for child in start_node.children.values():
            self.prune_tree(child, validation_data)
        # Find error before pruning.
        original_predictions = self.predict(validation_data)
        if self.type == "classification":
            original_error = evaluate("classerr", original_predictions, validation_data, 
                                      self.target_col)
        else:
            original_error = evaluate("mse", original_predictions, validation_data, 
                                      self.target_col)
        # Write the number of nodes in the tree and error to an output file for analysis.
        with open(self.output_file, mode = 'a') as file:
           file.write(str(self.count_nodes(self.root)) + "," + str(original_error) + "\n")
        # Test replacing the decision node with the pruned prediction. Convert it to a leaf node.
        original_feature = start_node.feature
        original_numeric_split = start_node.numeric_split
        original_children = start_node.children
        original_pruned_prediction = start_node.pruned_prediction
        original_prediction = start_node.prediction

        start_node.prediction = start_node.pruned_prediction
        start_node.feature = None
        start_node.numeric_split = None
        start_node.children = None
        start_node.pruned_prediction = None
        new_predictions = self.predict(validation_data)
        if self.type == "classification":
            new_error = evaluate("classerr", new_predictions, validation_data, self.target_col)
        else:
            new_error = evaluate("mse", new_predictions, validation_data, self.target_col)
        with open(self.output_file, mode = 'a') as file:
           file.write(str(self.count_nodes(self.root)) + "," + str(new_error) + "\n")
        # If pruning results in worse performance, undo the changes to the node.
        if new_error > original_error: 
            start_node.feature = original_feature
            start_node.numeric_split = original_numeric_split
            start_node.children = original_children
            start_node.pruned_prediction = original_pruned_prediction
            start_node.prediction = original_prediction