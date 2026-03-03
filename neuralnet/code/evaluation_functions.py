# evaluation_functions.py
# This file defines a function for evaluating the machine learning algorithms. It can compute 
# classification error and mean squared error.

import pandas as pd

def evaluate(type, predictions, test_data, target_col):
    """This function computes the error of a model. It can calculate classification error and mean 
    squared error (MSE).

    Args:
        type (str): The type of error to calculate. It can be either "mse" or "classerr".
        predictions (pd.Series): The predictions outputted by a model.
        test_data (pd.DataFrame): The test data.
        target_col (list): The name of the target column(s) in test_data, as a list of strings. For 
        the car data, this contains multiple values.

    Returns:
        (float): The classification error or MSE.
    """
    if type != "mse" and type != "classerr":
        raise Exception("type must be 'mse' or 'classerr'.")
    if len(target_col) != 1 and type == "mse":
        raise Exception("target_col must be length 1 when computing mse.")
    n_total_predictions = len(predictions)
    n_test_obs = len(test_data)
    if n_total_predictions != n_test_obs:
        raise Exception("The number of predictions and number of test observations must be equal.")
    if type == "mse":
        # Mean squared error = (sum of squared errors)/(# total predictions)
        total_squared_error = 0
        for index, value in test_data[target_col[0]].items():
            total_squared_error += (value - predictions[index])**2
        error = total_squared_error/n_total_predictions
    else:
        # If there are multiple target columns, the predictions returned by the model objects are
        # lists combinining one-hot encodings.
        if len(target_col) > 1:
            test_data_target_col = pd.Series(test_data[target_col].values.tolist()).reset_index(drop = True)
        else:
            test_data_target_col = test_data[target_col[0]]
        # Classification error = (# of incorrect predictions)/(# total predictions)
        comparison = test_data_target_col == predictions
        n_incorrect = n_total_predictions - comparison.sum()
        error = n_incorrect/n_total_predictions
    return error