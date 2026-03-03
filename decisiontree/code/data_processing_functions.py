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
        in the forestfires dataset). Categorical features have a "_cat" suffix, and numeric 
        features have a "_num" suffix. This allows the decision tree algorithm to easily keep track 
        of feature types.
    """
    # The forestfires dataset contains column headers, so it must be loaded differently.
    if "forestfires" in file_path:
        input_data = pd.read_csv(file_path)
        # Rename columns and add suffixes.
        input_data.columns = ["x_num", "y_num", "month_cat", "day_cat", "ffmc_num", "dmc_num", 
        "dc_num", "isi_num", "temp_num", "rh_num", "wind_num", "rain_num", "area"]
        # Log transform the area column.
        input_data["area"] = np.log(input_data["area"] + 1)
    # The other datasets do not contain column headers, so they can all be loaded the same way.
    else:
        input_data = pd.read_csv(file_path, header = None)
        # Add column names and drop columns if appropriate.
        if "breast-cancer-wisconsin" in file_path:
            input_data.columns = ["sample_code_number_cat", "clump_thickness_num" , 
            "uniformity_of_cell_size_num", "uniformity_of_cell_shape_num", "marginal_adhesion_num", 
            "single_epithelial_cell_size_num", "bare_nuclei_num", "bland_chromatin_num", 
            "normal_nucleoli_num", "mitoses_num", "class"]
            input_data.drop(columns = "sample_code_number_cat", inplace = True)
        elif "car" in file_path:
            input_data.columns = ["buying_cat", "maint_cat", "doors_cat", "persons_cat", 
            "lug_boot_cat", "safety_cat", "class"]
        elif "house-votes-84" in file_path:
            input_data.columns = ["class", "handicapped-infants_cat", 
            "water-project-cost-sharing_cat", "adoption-of-the-budget-resolution_cat", 
            "physician-fee-freeze_cat", "el-salvador-aid_cat", "religious-groups-in-schools_cat", 
            "anti-satellite-test-ban_cat", "aid-to-nicaraguan-contras_cat", "mx-missile_cat", 
            "immigration_cat", "synfuels-corporation-cutback_cat", "education-spending_cat", 
            "superfund-right-to-sue_cat", "crime_cat", "duty-free-exports_cat", 
            "export-administration-act-south-africa_cat"]
            # Recode "?" to "a" (abstain).
            input_data = input_data.replace("?", "a")
        elif "abalone" in file_path:
            input_data.columns = ["sex_cat", "length_num", "diameter_num", "height_num", 
            "whole_weight_num", "shucked_weight_num", "viscera_weight_num", "shell_weight_num", 
            "rings"]
        elif "machine" in file_path:
            input_data.columns = ["vendor_name_cat", "model_name_cat", "myct_num", "mmin_num", 
            "mmax_num", "cach_num", "chmin_num", "chmax_num", "prp", "erp_num"]
            input_data.drop(columns = ["vendor_name_cat", "model_name_cat"], inplace = True)
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
            input_data["bare_nuclei_num"] = input_data["bare_nuclei_num"].replace("?", 
                                                                                  imputed_value)
            input_data["bare_nuclei_num"] = pd.to_numeric(input_data["bare_nuclei_num"])
        else:
            input_data["bare_nuclei_num"] = input_data["bare_nuclei_num"].replace("?", np.nan)
            # Calculate the average of the column, ignoring NaNs.
            input_data["bare_nuclei_num"] = pd.to_numeric(input_data["bare_nuclei_num"], 
                                                          errors = "coerce")
            imputed_value = input_data["bare_nuclei_num"].mean()
            # Replace missing values with the average for the column.
            input_data["bare_nuclei_num"] = input_data["bare_nuclei_num"].replace(np.nan, 
                                                                              imputed_value)
    return input_data, imputed_value

def remove_misc_columns(input_data, dataset):
    """This function removes columns that should not be used as inputs in KNN. For example, the erp 
    column in the machine dataset.

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
        unwanted_cols = input_data[["erp_num"]]
        input_data.drop(columns = "erp_num", inplace = True)
    return input_data, unwanted_cols

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
    input_data, unwanted_cols = remove_misc_columns(input_data, dataset)
    return input_data, unwanted_cols