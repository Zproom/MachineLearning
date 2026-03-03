# experiments.py
# This file contains the experiments where I test the performance of regular decision trees against
# the performance of pruned decision trees and the null models.

import os
from null_model import *
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *
from decision_tree import *

def run_experiments(file_path, dataset, target_col, type, output_file):
    """This function runs a full experiment for a UCI dataset. It uses 5 x 2 cross-validation for 
    each model: (1) classification or regression tree, (2) pruned classification or regression
    tree, and (3) null model.

    Args:
        file_path (str): The full file path to the UCI data file.
        dataset (str): The name of the UCI dataset. 
        target_col (str): The name of the target column.
        type (str): The type of experiment. It can be either "classification" or "regression".
        output_file (str): The file to write output to for analysis. This stores data on the number
        of nodes vs. the error during the pruning process.
    
    Returns:
        nothing. Prints the average result of the ten experiments for each model.
    """
    if type != "classification" and type != "regression":
        raise Exception("type must be 'classification' or 'regression'.")
    eval_type = "classerr" if type == "classification" else "mse"
    if os.path.exists(output_file):
        os.remove(output_file)
    print("Running experiments for the following dataset: " + dataset)
    # 1. Load and prepare data.
    input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
    validation_data, non_validation_data = random_split(input_data, "validation", random_state = 1)
    # 2. Run regular decision tree experiments.
    results = []
    for i in range(5):
        if type == "classification":
            training_data, test_data = stratify_classes(non_validation_data, target_col, 
                                                        random_state = i)
        else:
            training_data, test_data = random_split(non_validation_data, "testing", 
                                                    random_state = i)
        # Handle missing values. Use the imputed training value (mean of non-missing observations) to
        # impute missing test and validation data.
        training_data, imputed_val = handle_missing_values(training_data, dataset)
        test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
        validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]
        
        decision_tree_a = decision_tree(training_data, target_col, type, output_file)
        decision_tree_a.train()
        predictions = decision_tree_a.predict(test_data)
        performance_a = evaluate(eval_type, predictions, test_data, target_col)
        
        decision_tree_b = decision_tree(test_data, target_col, type, output_file)
        decision_tree_b.train()
        predictions = decision_tree_b.predict(training_data)
        performance_b = evaluate(eval_type, predictions, training_data, target_col)
        results.append(performance_a)
        results.append(performance_b)
    print("Average Decision Tree Error: " + str(np.mean(results)))
    # 3. Run pruned tree experiments.
    results = []
    for i in range(5):
        if type == "classification":
            training_data, test_data = stratify_classes(non_validation_data, target_col, 
                                                        random_state = i)
        else:
            training_data, test_data = random_split(non_validation_data, "testing", 
                                                    random_state = i)
        # Handle missing values. Use the imputed training value (mean of non-missing observations) to
        # impute missing test and validation data.
        training_data, imputed_val = handle_missing_values(training_data, dataset)
        test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
        validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]
        
        decision_tree_a = decision_tree(training_data, target_col, type, output_file)
        decision_tree_a.train()
        decision_tree_a.prune(validation_data)
        predictions = decision_tree_a.predict(test_data)
        performance_a = evaluate(eval_type, predictions, test_data, target_col)
        
        decision_tree_b = decision_tree(test_data, target_col, type, output_file)
        decision_tree_b.train()
        decision_tree_b.prune(validation_data)
        predictions = decision_tree_b.predict(training_data)
        performance_b = evaluate(eval_type, predictions, training_data, target_col)
        results.append(performance_a)
        results.append(performance_b)
    print("Average Pruned Decision Tree Error: " + str(np.mean(results)))
    # 4. Run null model experiments
    results = []
    for i in range(5):
        if type == "classification":
            training_data, test_data = stratify_classes(non_validation_data, target_col,
                                                        random_state = i)
        else:
            training_data, test_data = random_split(non_validation_data, "testing",
                                                    random_state = i)
        # Handle missing values. Use the imputed training value (mean of non-missing observations) to
        # impute missing test and validation data.
        training_data, imputed_val = handle_missing_values(training_data, dataset)
        test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
        validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]
        
        null_classifier_a = null_model(training_data[target_col], type)
        predictions = null_classifier_a.predict(len(test_data))
        performance_a = evaluate(eval_type, predictions, test_data, target_col)
        
        null_classifier_b = null_model(test_data[target_col], type)
        predictions = null_classifier_b.predict(len(training_data))
        performance_b = evaluate(eval_type, predictions, training_data, target_col)
        results.append(performance_a)
        results.append(performance_b)
    print("Average Null Model Error: " + str(np.mean(results)))
    # Evaluate the authors' predictions (ERP).
    if dataset == "machine":
        input_data = pd.concat([input_data, excluded_cols], axis = 1)
        validation_data, non_validation_data = random_split(input_data, "validation", 
                                                            random_state = 1)
        results = []
        for i in range(5):
            if type == "classification":
                training_data, test_data = stratify_classes(non_validation_data, target_col,
                                                            random_state = i)
            else:
                training_data, test_data = random_split(non_validation_data, "testing",
                                                        random_state = i)
            # Handle missing values. Use the imputed training value (mean of non-missing observations) to
            # impute missing test and validation data.
            training_data, imputed_val = handle_missing_values(training_data, dataset)
            test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
            validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]
            
            performance_a = evaluate(eval_type, test_data["erp_num"], test_data, target_col)
            performance_b = evaluate(eval_type, training_data["erp_num"], training_data, target_col)
            results.append(performance_a)
            results.append(performance_b)
    print("Average Author Prediction Error: " + str(np.mean(results)))
    print("Finished experiment!")
    print("")

###################################################################################################
# Cancer data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = "class"
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment2/data/breast-cancer-wisconsin_prune.csv"
# 2. Run experiments.
run_experiments(file_path, dataset, target_col, type, output_file)

###################################################################################################
# Car data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/car.data"
dataset = "car"
target_col = "class"
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment2/data/car_prune.csv"
# 2. Run experiments.
run_experiments(file_path, dataset, target_col, type, output_file)

###################################################################################################
# Voting data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/house-votes-84.data"
dataset = "house-votes-84"
target_col = "class"
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment2/data/house-votes-84_prune.csv"
# 2. Run experiments.
run_experiments(file_path, dataset, target_col, type, output_file)

###################################################################################################
# Abalone data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/abalone.data"
dataset = "abalone"
target_col = "rings"
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment2/data/abalone_prune.csv"
# 2. Run experiments.
run_experiments(file_path, dataset, target_col, type, output_file)

###################################################################################################
# Computer data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/machine.data"
dataset = "machine"
target_col = "prp"
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment2/data/machine_prune.csv"
# 2. Run experiments.
run_experiments(file_path, dataset, target_col, type, output_file)

###################################################################################################
# Forest Fire data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/forestfires.csv"
dataset = "forestfires"
target_col = "area"
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment2/data/forestfires_prune.csv"
# 2. Run experiments.
run_experiments(file_path, dataset, target_col, type, output_file)