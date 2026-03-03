# cross_validation_functions.py
# This file contains functions to assist with cross validation. There is a function to randomly
# split input data and a function to stratify classes.

import numpy as np
import pandas as pd
from data_processing_functions import *

def random_split(input_data, type, random_state):
    """This function randomly splits a dataframe. This can be used to split off a validation set
    or to split the non-validation data in half for regression.

    Args:
        input_data (pd.DataFrame): The input data to be split.
        type (str): The type of split. If you're using the function to split off a validation set,
        set this to "validation". If you're using the function to split training and test data, set
        this to "testing".
        random_state (int): The random seed to use. Ensures the results are reproducible.

    Returns:
        (tuple): A two-element tuple containing:
            If type == "validation":
                (pd.DataFrame): Validation set.
                (pd.DataFrame): Non-validation set (training plus test sets).
            If type == "testing":
                (pd.DataFrame): Test set.
                (pd.DataFrame): Training set.
    """
    if type != "validation" and type != "testing":
        raise Exception("type must be set to 'validation' or 'testing'.")
    # Randomly split the data into two smaller dataframes. The validation set should contain 20% of 
    # the original data, and the non-validation set should contain 80% of the original data.
    if type == "validation":
        # Set a seed for reproducibility.
        validation_data = input_data.sample(frac = 0.2, random_state = random_state)
        non_validation_data = input_data.drop(validation_data.index)
        validation_data = validation_data.reset_index(drop = True)
        non_validation_data = non_validation_data.reset_index(drop = True)
        return validation_data, non_validation_data
    # Randomly split the original data into two halves.
    else:
        test_data = input_data.sample(frac = 0.5, random_state = random_state)
        training_data = input_data.drop(test_data.index)
        test_data = test_data.reset_index(drop = True)
        training_data = training_data.reset_index(drop = True)
        return test_data, training_data

def stratify_classes(input_data, target_col, random_state):
    """This function randomly splits a dataframe in half, and it stratifies the classes. This
    should be used for classification problems.

    Args:
       input_data (pd.DataFrame): The input data to be split.
       target_col (list): The name of the target column(s) in test_data, as a list of strings. 
       There may be multiple target columns. For example, if the target value takes on more than 
       two unique values (car evaluation dataset).
       random_state (int): The random seed to use. Ensures the results are reproducible.

    Returns:
        (tuple): A two-element tuple containing:
            If type == "testing":
                (pd.DataFrame): Test set.
                (pd.DataFrame): Training set.
    """
    # Combine the target columns into one column if needed.
    if len(target_col) > 1:
        input_data["label_tmp"] = pd.Series(input_data[target_col].values.tolist())
    elif len(target_col) == 1:
        input_data["label_tmp"] = input_data[target_col[0]]
    else:
        raise Exception("target_col cannot be empty.")
    # Find unique class labels.
    if len(target_col) > 1:
        stratify_labels = input_data["label_tmp"].drop_duplicates()
    else:
        stratify_labels = input_data["label_tmp"].unique()
    test_data, training_data = [], []
    # Stratify by each class label and shuffle rows within each class.
    for label in stratify_labels:
        if len(target_col) > 1:
            class_df_ind = input_data["label_tmp"].apply(lambda x: x == label)
            class_df = input_data[class_df_ind]
        else:
            class_df = input_data[input_data["label_tmp"] == label]
        test_data_class = class_df.sample(frac = 0.5, random_state = random_state)
        training_data_class = class_df.drop(test_data_class.index)
        test_data.append(test_data_class)
        training_data.append(training_data_class)
    # Combine the shuffled DataFrames and shuffle again.
    test_data = pd.concat(test_data).sample(frac = 1, random_state = 
                                            random_state).reset_index(drop = True)
    training_data = pd.concat(training_data).sample(frac = 1, random_state = 
                                                    random_state).reset_index(drop = True)
    # Remove temporary target column.
    test_data.drop(columns = ["label_tmp"], inplace = True)
    training_data.drop(columns = ["label_tmp"], inplace = True)
    return test_data, training_data