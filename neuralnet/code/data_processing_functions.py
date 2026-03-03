# data_processing_functions.py
# This file defines data processing functions (machine learning pipeline) that prepare data sets 
# for the machine learning algorithms.

import pandas as pd
import numpy as np

def load_data(file_path):
    """This function loads the .csv files on disk into pandas Dataframes.

    Args:
        file_path (str): File path to the .csv file.

    Returns:
        (pd.DataFrame): The input data, with a few transformations (add/rename columns, drop
        certain columns, recode "?" to "a" in voting dataset, and log transform the area column 
        in the forestfires dataset).
    """
    # The forestfires dataset contains column headers, so it must be loaded differently.
    if "forestfires" in file_path:
        input_data = pd.read_csv(file_path)
        # Convert column names to lower case for consistency.
        input_data.columns = [s.lower() for s in input_data.columns]
        # Log transform the area column.
        input_data["area"] = np.log(input_data["area"] + 1)
    # The other datasets do not contain column headers, so they can all be loaded the same way.
    else:
        input_data = pd.read_csv(file_path, header = None)
        # Add column names and drop columns if appropriate.
        if "breast-cancer-wisconsin" in file_path:
            input_data.columns = ["sample_code_number", "clump_thickness" , 
            "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion", 
            "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", 
            "mitoses", "class"]
            input_data.drop(columns = "sample_code_number", inplace = True)
        elif "car" in file_path:
            input_data.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", 
            "class"]
        elif "house-votes-84" in file_path:
            input_data.columns = ["class", "handicapped-infants", "water-project-cost-sharing",
            "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid",
            "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras",
            "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", 
            "superfund-right-to-sue", "crime", "duty-free-exports", 
            "export-administration-act-south-africa"]
            # Recode "?" to "a" (abstain).
            input_data = input_data.replace("?", "a")
        elif "abalone" in file_path:
            input_data.columns = ["sex", "length", "diameter", "height", "whole_weight", 
            "shucked_weight", "viscera_weight", "shell_weight", "rings"]
        elif "machine" in file_path:
            input_data.columns = ["vendor_name", "model_name", "myct", "mmin", "mmax", "cach", 
            "chmin", "chmax", "prp", "erp"]
            input_data.drop(columns = ["vendor_name", "model_name"], inplace = True)
    return input_data

def handle_missing_values(input_data, dataset, imputed_value = None):
    """This function recodes missing data.

    Args:
        input_data (pd.DataFrame): The input data.
        dataset (str): The name of the UCI dataset.
        imputed_value (float): If provided, use this value to replace missing values. By default,
        this argument is set to None, and the average of the non-missing values in the same column
        is used to replace missing values.

    Returns:
        (tuple): A two-element tuple containing:
            (pd.DataFrame): The input data, with missing values recoded.
            (float): The value used to impute missing observations.
    """
    if dataset == "breast-cancer-wisconsin":
        if imputed_value is not None:
            input_data["bare_nuclei"] = input_data["bare_nuclei"].replace("?", imputed_value)
            input_data["bare_nuclei"] = pd.to_numeric(input_data["bare_nuclei"])
        else:
            input_data["bare_nuclei"] = input_data["bare_nuclei"].replace("?", np.nan)
            # Calculate the average of the column, ignoring NaNs.
            input_data["bare_nuclei"] = pd.to_numeric(input_data["bare_nuclei"], errors = "coerce")
            imputed_value = input_data["bare_nuclei"].mean()
            # Replace missing values with the average for the column.
            input_data["bare_nuclei"] = input_data["bare_nuclei"].replace(np.nan, imputed_value)
    return input_data, imputed_value

def handle_categorical_data(input_data, dataset):
    """This function creates one-hot encodings for nominal data and label encodings for ordinal 
    data.

    Args:
        input_data (pd.DataFrame): The input data.
        dataset (str): The name of the UCI dataset.

    Returns:
        (pd.DataFrame): The input data, with categorical data recoded.
    """
    # Create one-hot encodings for nominal data.
    if dataset == "breast-cancer-wisconsin":
        one_hot_encoding_df = pd.get_dummies(input_data["class"], prefix = "class", dtype = int, 
        drop_first = True) # Set drop_first = True to prevent adding redundant columns.
        input_data = pd.concat([input_data, one_hot_encoding_df], axis = 1)
        input_data.drop(columns = "class", inplace = True)
    elif dataset == "car":
        one_hot_encoding_features_df = pd.get_dummies(input_data.drop(columns = "class"), 
        dtype = int, drop_first = True)  
        one_hot_encoding_target_df = pd.get_dummies(input_data[["class"]], dtype = int)
        input_data = pd.concat([one_hot_encoding_features_df, one_hot_encoding_target_df], 
                               axis = 1)
    elif dataset == "house-votes-84":
        one_hot_encoding_df = pd.get_dummies(input_data, dtype = int, drop_first = True)  
        input_data = one_hot_encoding_df
    elif dataset == "abalone":
        one_hot_encoding_df = pd.get_dummies(input_data["sex"], prefix = "sex", dtype = int, 
        drop_first = True)
        input_data = pd.concat([input_data, one_hot_encoding_df], axis = 1)
        input_data.drop(columns = "sex", inplace = True)
    # Create label encodings for ordinal data.
    elif dataset == "forestfires":
        input_data = input_data.replace({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, 
        "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12, "mon": 1, 
        "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7})
    return input_data

def remove_misc_columns(input_data, dataset):
    """This function removes columns that should not be used as inputs in the machine learning 
    algorithms. For example, the erp column in the machine dataset.

    Args:
        input_data (pd.DataFrame): The input data.
        dataset (str): The name of the UCI dataset.

    Returns:
        (tuple): A two-element tuple containing:
            (pd.DataFrame): The input data, with unwanted columns removed.
            (pd.DataFrame): The unwanted columns. If there are no unwanted columns, the function 
            returns an empty string.
    """
    unwanted_cols = ""
    if dataset == "machine":
        # Remove the erp column because it was estimated by the authors using a linear regression 
        # method. Return it as a separate object for later comparisons.
        unwanted_cols = input_data[["erp"]]
        input_data.drop(columns = "erp", inplace = True)
    return input_data, unwanted_cols

def normalize_data(input_data, dataset):
    """This function normalizes numeric features using z-score normalization. The target column is 
    not normalized.

    Args:
        input_data (pd.DataFrame): The input data.
        dataset (str): The name of the UCI dataset.
    
    Returns:
        (pd.DataFrame): The input data, with numeric feature columns normalized.
    """
    if dataset == "breast-cancer-wisconsin":
        numeric_columns = [col for col in input_data.columns if col not in ["class_2", "class_4"]]
    elif dataset == "abalone":
        numeric_columns = [col for col in input_data.columns if col not in ["sex_F", "sex_I", 
        "sex_M", "rings"]]
    elif dataset == "machine":
        numeric_columns = [col for col in input_data.columns if col not in ["prp"]]
    elif dataset == "forestfires":
        numeric_columns = [col for col in input_data.columns if col not in ["area"]]
    else: # Don't normalize features in other datasets.
        return input_data
    for col in numeric_columns:
        input_data[col] = (input_data[col] - input_data[col].mean()) / (input_data[col].std())
    return input_data

def run_data_loading_pipeline(file_path, dataset):
    """This function runs all the data processing functions (machine learning pipeline) on an input
    dataset to prepare it for the k-nearest neighbor algorithms.

    Args:
        file_path (str): File path to the .csv file.
        dataset (str): The name of the UCI dataset.

    Returns:
        (pd.Dataframe): Input data ready to be used by the k-nearest neighbor algorithms.
        (pd.DataFrame): Miscellaneous columns that should not be used as features in the k-nearest 
        neighbor algorithms, but might be useful for later comparisons (e.g. the erp column in the 
        machine dataset).
    """
    if dataset not in file_path:
        raise Exception("dataset must be a substring in file_path.")
    input_data = load_data(file_path)
    input_data = handle_categorical_data(input_data, dataset)
    input_data, unwanted_cols = remove_misc_columns(input_data, dataset)
    return input_data, unwanted_cols