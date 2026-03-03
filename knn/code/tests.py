# tests.py
# This file contains tests to ensure the knn methods and helper functions work properly.

from regular_knn import *
from edited_knn import *
from null_model import *
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *

# Test loading data.
input_data, excluded_cols = run_data_loading_pipeline("/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin.data",
                                                      "breast-cancer-wisconsin")
validation_data, other_data = random_split(input_data, "validation")
test_data, training_data = stratify_classes(other_data, ["class_4"])


# # Tests for regular KNN
# ## Load data
# file_path = "/home/zacharyproom/IntroML/assignment1/data/house-votes-84.data"
# dataset = "house-votes-84"
# target_col = ["class_republican"]
# x = load_data(file_path)
# x = handle_missing_values(x, dataset)
# x = handle_categorical_data(x, dataset)
# x = remove_misc_columns(x, dataset)
# x = normalize_data(x, dataset)
# print(x)

# # Run KNN Classification
# ## Split input data in half. Implement proper cross validation later.
# n = len(x)
# midpoint = round(n/2)
# test_data = x.iloc[0:midpoint]
# training_data = x.iloc[midpoint:n].reset_index(drop = True)
# classifier = knn_classifier(2, training_data)
# predictions = classifier.predict(test_data, target_col)
# print(predictions.to_string())
# print(evaluate("classerr", predictions, test_data, target_col))
# nullmodel = null_model(training_data[target_col], "classification")
# null_predictions = nullmodel.predict(len(test_data))
# print(evaluate("classerr", null_predictions, test_data, target_col)) # Null model error.

# ## Load data
# file_path = "/home/zacharyproom/IntroML/assignment1/data/forestfires.csv"
# dataset = "forestfires"
# target_col = ["area"]
# x = load_data(file_path)
# x = handle_missing_values(x, dataset)
# x = handle_categorical_data(x, dataset)
# x = remove_misc_columns(x, dataset)
# x = normalize_data(x, dataset)
# print(x)

# # Run KNN Regression
# ## Split input data in half. Implement proper cross validation later.
# n = len(x)
# midpoint = round(n/2)
# test_data = x.iloc[0:midpoint]
# training_data = x.iloc[midpoint:n].reset_index(drop = True)
# regressor = knn_regressor(3, 10, training_data)
# predictions = regressor.predict(test_data, target_col)
# print(predictions.to_string())
# print(evaluate("mse", predictions, test_data, target_col))
# nullmodel = null_model(training_data[target_col], "regression")
# null_predictions = nullmodel.predict(len(test_data))
# print(evaluate("mse", null_predictions, test_data, target_col)) # Null model error.

# Tests for edited KNN
## Load data
# file_path = "/home/zacharyproom/IntroML/assignment1/data/house-votes-84.data"
# dataset = "house-votes-84"
# target_col = ["class_republican"]
# x = load_data(file_path)
# x = handle_missing_values(x, dataset)
# x = handle_categorical_data(x, dataset)
# x = remove_misc_columns(x, dataset)
# x = normalize_data(x, dataset)
# print(x)

# # Run KNN Classification
# ## Split input data in half. Implement proper cross validation later.
# n = len(x)
# onethirdpt = round(n/3)
# twothirdpt = round(2*n/3)
# test_data = x.iloc[0:onethirdpt]
# validation_data = x.iloc[onethirdpt:twothirdpt].reset_index(drop = True)
# training_data = x.iloc[twothirdpt:n].reset_index(drop = True)
# classifier = edited_knn_classifier(2, training_data)
# classifier.train(validation_data, target_col)
# predictions = classifier.predict(test_data, target_col)
# print(predictions.to_string())
# print(evaluate("classerr", predictions, test_data, target_col))
# nullmodel = null_model(training_data[target_col], "classification")
# null_predictions = nullmodel.predict(len(test_data))
# print(evaluate("classerr", null_predictions, test_data, target_col)) # Null model error.

# ## Load data
# file_path = "/home/zacharyproom/IntroML/assignment1/data/forestfires.csv"
# dataset = "forestfires"
# target_col = ["area"]
# x = load_data(file_path)
# x = handle_missing_values(x, dataset)
# x = handle_categorical_data(x, dataset)
# x = remove_misc_columns(x, dataset)
# x = normalize_data(x, dataset)
# print(x)

# # Run KNN Regression
# ## Split input data in half. Implement proper cross validation later.
# n = len(x)
# onethirdpt = round(n/3)
# twothirdpt = round(2*n/3)
# test_data = x.iloc[0:onethirdpt]
# validation_data = x.iloc[onethirdpt:twothirdpt].reset_index(drop = True)
# training_data = x.iloc[twothirdpt:n].reset_index(drop = True)
# regressor = edited_knn_regressor(3, 10, 0.5, training_data)
# regressor.train(validation_data, target_col)
# predictions = regressor.predict(test_data, target_col)
# print(predictions.to_string())
# print(evaluate("mse", predictions, test_data, target_col))
# nullmodel = null_model(training_data[target_col], "regression")
# null_predictions = nullmodel.predict(len(test_data))
# print(evaluate("mse", null_predictions, test_data, target_col)) # Null model error.