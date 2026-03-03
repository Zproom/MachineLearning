# training_helper_functions.py
# These functions are used across the decision tree classes to help with training.

import numpy as np
import pandas as pd
from evaluation_functions import evaluate

def split_num(input_data, feature, split_value):
    """This function splits input data into two subsets using a feature and a split value. It
    should only be applied to numeric features.

    Args:
        input_data (pd.DataFrame): The data in the tree partition.
        feature (str): The name of the feature being used for splitting.
        split_value (float): The feature value to use to split the data into two subsets.

    Returns:
        (tuple): A two-element tuple containing:
            (pd.DataFrame): The data below (and including) the split value.
            (pd.DataFrame): The data above the split value.
    """
    subset_below = input_data[input_data[feature] <= split_value].reset_index(drop = True)
    subset_above = input_data[input_data[feature] > split_value].reset_index(drop = True)
    return subset_below, subset_above

def split_cat(input_data, feature):
    """This function splits input data into subsets where each subset corresponds to a unique
    feature value. It should only be applied to categorical features.

    Args:
        input_data (pd.DataFrame): The data in the tree partition.
        feature (str): The name of the feature being used for splitting.

    Returns:
        (tuple): A two-element tuple containing:
            (list): A list of subsets, where each subset corresponds to a unique feature value. The
            feature values are sorted alphabetically, and the ordering of the subsets matches this.
            (list): The unique feature values.
    """
    feature_values = np.unique(input_data[feature]).tolist()
    subsets = []
    for val in feature_values:
        subset = input_data[input_data[feature] == val].reset_index(drop = True)
        subsets.append(subset)
    return subsets, feature_values

def calculate_entropy(input_data, target_col):
    """This function calculates the entropy of input data.
    
    Args:
        input_data (pd.DataFrame): The data in the tree partition.
        target_col (str): The name of the target column.

    Returns:
        (float): The entropy of the input data.
    """
    classes, class_counts = np.unique(input_data[target_col], return_counts = True)
    class_probabilities = class_counts/class_counts.sum()
    # Avoid log(0).
    class_probabilities = class_probabilities[class_probabilities > 0]  
    # Entropy = -1 * Sum Over All Classes(class probability * log2(class probability))
    return -np.sum(class_probabilities * np.log2(class_probabilities))

def calculate_gain_ratio_cat(input_data, target_col, feature, entropy):
    """This function calculates the gain ratio for a categorical feature.

    Args:
        input_data (pd.DataFrame): The data in the tree partition.
        target_col (str): The name of the target column.
        feature (str): The name of the feature being used for splitting.
        entropy (float): The entropy of the input data.

    Returns:
        (float): The information gain ratio.
    """
    # Calculate information gain.
    feature_values, feature_value_counts = np.unique(input_data[feature], return_counts = True)
    expected_entropy = 0
    for i in range(len(feature_values)):
        # Subset of the input data with the feature value.
        input_data_feature_value = input_data[input_data[feature] == feature_values[i]]
        entropy_feature_value = calculate_entropy(input_data_feature_value, target_col)
        expected_entropy += feature_value_counts[i]/len(input_data) * entropy_feature_value
    information_gain = entropy - expected_entropy
    # Calculate split information (IV).
    feature_value_probabilities = feature_value_counts/len(input_data)
    # Avoid log(0).
    feature_value_probabilities = feature_value_probabilities[feature_value_probabilities > 0]  
    split_information = -np.sum(feature_value_probabilities * np.log2(feature_value_probabilities))
    # If split_information is 0 (all data fall into one partition), return 0.
    if split_information == 0:
        return 0
    else:
        # Calculate the gain ratio: (information gain)/IV.
        return information_gain/split_information

def calculate_gain_ratio_num(input_data, target_col, feature, split_value, entropy):
    """This function calculates the gain ratio for a numeric feature.

    Args:
        input_data (pd.DataFrame): The data in the tree partition.
        target_col (str): The name of the target column.
        feature (str): The name of the feature being used for splitting.
        split_value (float): The feature value to use to split the data into two subsets.
        entropy (float): The entropy of the input data.

    Returns:
        (float): The information gain ratio.
    """
    # Calculate information gain.
    subset_below, subset_above = split_num(input_data, feature, split_value)
    subset_below_entropy = calculate_entropy(subset_below, target_col)
    subset_above_entropy = calculate_entropy(subset_above, target_col)
    n_subset_below = len(subset_below)
    n_subset_above = len(subset_above)
    expected_entropy = n_subset_below/len(input_data) * subset_below_entropy + \
    n_subset_above/len(input_data) * subset_above_entropy
    information_gain = entropy - expected_entropy
    # Calculate split information (IV).
    subset_probabilities = np.array([n_subset_below, n_subset_above])/len(input_data)
    # Avoid log(0).
    subset_probabilities = subset_probabilities[subset_probabilities > 0]  
    split_information = -np.sum(subset_probabilities * np.log2(subset_probabilities))
    # If split_information is 0 (all data fall into one partition), return 0.
    if split_information == 0:
        return 0
    else:
        # Calculate the gain ratio: (information gain)/IV.
        return information_gain/split_information
    
def calculate_mse_cat(input_data, target_col, feature):
    """This function calculates the mean squared error (MSE) for a categorical feature.

    Args:
        input_data (pd.DataFrame): The data in the tree partition.
        target_col (str): The name of the target column.
        feature (str): The name of the feature being used for splitting.

    Returns:
        (float): The MSE.
    """
    # Calculate information gain.
    feature_values = np.unique(input_data[feature])
    total_squared_error = 0
    for i in range(len(feature_values)):
        # Subset of the input data with the feature value.
        input_data_feature_value = input_data[input_data[feature] == feature_values[i]]
        prediction = input_data_feature_value[target_col].mean()
        predictions = pd.Series(np.repeat(prediction, 
                                          len(input_data_feature_value))).reset_index(drop = True)
        actual_values = input_data_feature_value[target_col]
        tss_feature_value = ((actual_values - predictions)**2).sum()
        total_squared_error += tss_feature_value
    return total_squared_error/len(input_data)

def calculate_mse_num(input_data, target_col, feature, split_value):
    """This function calculates the mean squared error (MSE) for a numeric feature.

    Args:
        input_data (pd.DataFrame): The data in the tree partition.
        target_col (str): The name of the target column.
        feature (str): The name of the feature being used for splitting.
        split_value (float): The feature value to use to split the data into two subsets.

    Returns:
        (float): The MSE.
    """
    # Calculate information gain.
    subset_below, subset_above = split_num(input_data, feature, split_value)
    total_squared_error = 0
    for subset in [subset_below, subset_above]:
        prediction = subset[target_col].mean()
        predictions = pd.Series(np.repeat(prediction, len(subset))).reset_index(drop = True)
        actual_values = subset[target_col]
        tss_subset = ((actual_values - predictions)**2).sum()
        total_squared_error += tss_subset
    return total_squared_error/len(input_data)