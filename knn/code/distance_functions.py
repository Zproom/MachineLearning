# distance_functions.py
# This file defines distance functions for KNN. They handle both categorical and numeric data.

import pandas as pd
import numpy as np

def distance_num(x, y, cyclical = False):
    """This function computes Euclidean distances between an observation and a DataFrame.

    Args:
        x (pandas.Series): First data point (row).
        y (pandas.DataFrame): DataFrame of observations to compute distances against.
        cyclical (bool): The data include cyclical columns (only relevant for forest fires data).
        By default, this is set to False.
    
    Returns:
        (pd.Series): The Euclidean distances between x and y. 
    """
    n = len(y)
    # Create a DataFrame of length n with x repeated. This will make it easier to calculate
    # distances efficiently.
    data_x = pd.DataFrame([x])
    data_x = pd.concat([data_x] * n, ignore_index = True)
    if cyclical:
        squared_differences = 0
        non_cyclical_data_x = data_x.drop(columns = ["month", "day"])
        month_diffs = []
        day_diffs = []
        for index, row in data_x.iterrows():
            data_x_month = data_x.loc[index, "month"]
            y_month = y.loc[index, "month"]
            data_x_day = data_x.loc[index, "day"]
            y_day = y.loc[index, "day"]
            if data_x_month == y_month:
                month_diff = 0
            else:
                month_diff = np.min([(data_x_month - y_month) % 1 + 1/11, 
                                     (y_month - data_x_month) % 1 + 1/11])
            if data_x_day == y_day:
                day_diff = 0
            else:
                day_diff = np.min([(data_x_day - y_day) % 1 + 1/6, 
                                   (y_day - data_x_day) % 1 + 1/6])
            month_diffs.append(month_diff)
            day_diffs.append(day_diff)
        differences_cyclical = pd.DataFrame({"month": month_diffs, "day": day_diffs})
        differences_non_cyclical = non_cyclical_data_x - y.drop(columns = ["month", "day"])
        differences_all = pd.concat([differences_non_cyclical, differences_cyclical], axis = 1)
        squared_differences = differences_all**2
    else:
        # Calculate squared differences for each column.
        squared_differences = (data_x - y)**2
    # Take the sum of the squared differences and sum across columns.
    total_sum_of_squares = squared_differences.sum(axis = 1)
    return total_sum_of_squares**0.5