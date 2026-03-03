# tests.py
# This file contains tests to ensure the decision tree methods and helper functions work properly.

from null_model import *
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *
from decision_tree import *

###################################################################################################
# Cancer data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = "class"
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin_results.csv"

# 2. Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")
training_data, test_data = stratify_classes(non_validation_data, target_col)
# Handle missing values. Use the imputed training value (mean of non-missing observations) to
# impute missing test and validation data.
training_data, imputed_val = handle_missing_values(training_data, dataset)
test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]

# 3. Train the classification tree.
classification_tree = decision_tree(training_data, target_col, type)
classification_tree.train()
predictions = classification_tree.predict(test_data)
peformance = evaluate("classerr", predictions, test_data, target_col)
print("Classification Tree Error: " + str(peformance))

# 4. Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions = null_classifier.predict(len(test_data))
peformance = evaluate("classerr", predictions, test_data, target_col)
print("Null Model Error: " + str(peformance))

# 5. Prune the tree.
classification_tree.prune(validation_data)
predictions = classification_tree.predict(test_data)
peformance = evaluate("classerr", predictions, test_data, target_col)
print("Pruned Classification Tree Error: " + str(peformance))

###################################################################################################
# Computer data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/machine.data"
dataset = "machine"
target_col = "prp"
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment1/data/machine_results.csv"

# 2. Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")
training_data, test_data = random_split(non_validation_data, "testing")
# Handle missing values. Use the imputed training value (mean of non-missing observations) to
# impute missing test and validation data.
training_data, imputed_val = handle_missing_values(training_data, dataset)
test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]

# 3. Train the classification tree.
regression_tree = decision_tree(test_data, target_col, type)
regression_tree.train()
predictions = regression_tree.predict(test_data)
peformance = evaluate("mse", predictions, test_data, target_col)
print("Regression Tree Error: " + str(peformance))

# 4. Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions = null_classifier.predict(len(test_data))
peformance = evaluate("mse", predictions, test_data, target_col)
print("Null Model Error: " + str(peformance))

# 5. Prune the tree.
regression_tree.prune(validation_data)
predictions = regression_tree.predict(test_data)
peformance = evaluate("mse", predictions, test_data, target_col)
print("Pruned Regression Tree Error: " + str(peformance))