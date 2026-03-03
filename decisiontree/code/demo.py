# demo.py
# This file contains demo code for the video.

from null_model import *
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *
from decision_tree import *

###################################################################################################
# Cancer data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = "class"
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment2/data/breast-cancer-wisconsin_prune.csv"

input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation", random_state = 1)

if type == "classification":
            training_data, test_data = stratify_classes(non_validation_data, target_col, 
                                                        random_state = 0)
else:
    training_data, test_data = random_split(non_validation_data, "testing", 
                                            random_state = 0)
# Handle missing values. Use the imputed training value (mean of non-missing observations) to
# impute missing test and validation data.
training_data, imputed_val = handle_missing_values(training_data, dataset)
test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]

decision_tree_a = decision_tree(training_data, target_col, type, output_file)
decision_tree_a.train()
# 1. Demonstrate an example traversing a classification tree and a class label being assigned at the
# leaf.
# 2. Show sample outputs from one test fold for a classification tree and one test fold for a 
# regression tree.
predictions = decision_tree_a.predict(test_data)
print(predictions)
# 3. Show a sample classification tree without pruning and a sample classification tree with pruning.
decision_tree_a.print_tree(decision_tree_a.root, 0)
print("")
print("")
decision_tree_a.prune(validation_data)
decision_tree_a.print_tree(decision_tree_a.root, 0)
print("")
print("")

###################################################################################################
# Computer data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment1/data/machine.data"
dataset = "machine"
target_col = "prp"
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment2/data/machine_prune.csv"

input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation", random_state = 1)

if type == "classification":
            training_data, test_data = stratify_classes(non_validation_data, target_col, 
                                                        random_state = 0)
else:
    training_data, test_data = random_split(non_validation_data, "testing", 
                                            random_state = 0)
# Handle missing values. Use the imputed training value (mean of non-missing observations) to
# impute missing test and validation data.
training_data, imputed_val = handle_missing_values(training_data, dataset)
test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]

decision_tree_a = decision_tree(training_data, target_col, type, output_file)
decision_tree_a.train()
# 4. Demonstrate an example traversing a regression tree and a prediction being made at the leaf.
predictions = decision_tree_a.predict(test_data)
# 5. Show a sample regression tree without pruning and a sample regression tree with pruning.
decision_tree_a.print_tree(decision_tree_a.root, 0)
print("")
print("")
decision_tree_a.prune(validation_data)
decision_tree_a.print_tree(decision_tree_a.root, 0)
print("")
print("")

###################################################################################################
# Cancer data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = "class"
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment2/data/breast-cancer-wisconsin_prune.csv"

input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation", random_state = 1)

if type == "classification":
            training_data, test_data = stratify_classes(non_validation_data, target_col, 
                                                        random_state = 0)
else:
    training_data, test_data = random_split(non_validation_data, "testing", 
                                            random_state = 0)
# Handle missing values. Use the imputed training value (mean of non-missing observations) to
# impute missing test and validation data.
training_data, imputed_val = handle_missing_values(training_data, dataset)
test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]

decision_tree_a = decision_tree(training_data, target_col, type, output_file)
# 6. Demonstrate the calculation of information gain, gain ratio, and mean squared error.
# Set breakpoint in calculate_gain_ratio_num() and then run the code below.
decision_tree_a.train()
# 7. Demonstrate a decision being made to prune a subtree.
# Uncomment code and set a breakpoint in prune_tree() and then run the code below.
decision_tree_a.prune(validation_data)
