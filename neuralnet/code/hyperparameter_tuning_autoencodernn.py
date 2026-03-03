# hyperparameter_tuning_autoencodernn.py
# This file is used to tune the number of hidden nodes in the neural networks before comparing all
# the models with 5 x 2 cross-validation. These are the autoencoder-based neural networks.

import os
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *
from null_model import *
from autoencoder_based_classifier import *
from autoencoder_based_regressor import *

def tune_hyperparameters(file_path, dataset, target_col, type, output_file, h1_list, h2_list):
    """This function performs hyperparameter tuning for the neural network models. It tunes the
    number of nodes in the first and second hidden layers.

    Args:
        file_path (str): The full file path to the UCI data file.
        dataset (str): The name of the UCI dataset. 
        target_col (list): The name of the target column(s).
        type (str): The type of experiment. It can be either "classification" or "regression".
        output_file (str): The file to write output to for analysis. This stores data on the number
        of nodes vs. the error during the pruning process.
        h1_list (list): The number of nodes in the first hidden layer to test.
        h2_list (list): The number of nodes in the second hidden layer to test.
    
    Returns:
        Prints the best performing hyperparameter settings and saves the results to the output file.
    """
    if type != "classification" and type != "regression":
        raise Exception("type must be 'classification' or 'regression'.")
    eval_type = "classerr" if type == "classification" else "mse"
    if os.path.exists(output_file):
        os.remove(output_file)
    print("Running experiments for the following dataset: " + dataset)
    all_parameter_settings = [(h1, h2) for h1 in h1_list for h2 in h2_list]
    results_df = pd.DataFrame(all_parameter_settings, columns = ["h1", "h2"])
    results_df[eval_type] = np.nan
    for index, row in results_df.iterrows():
        h1 = results_df.loc[index, "h1"]
        h2 = results_df.loc[index, "h2"]
        average_performances = []
        for i in range(5):
            # Data prep
            input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
            validation_data, non_validation_data = random_split(input_data, "validation", random_state = i)
            if type == "classification":
                training_data, test_data = stratify_classes(non_validation_data, target_col, random_state = i)
            else:
                training_data, test_data = random_split(non_validation_data, "testing", random_state = 1)
            # Handle missing values. Use the imputed training value (mean of non-missing observations) to
            # impute missing test and validation data.
            training_data, imputed_val = handle_missing_values(training_data, dataset)
            test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
            validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]
            # Normalize the data. Normalize the training, test, and validation separately to prevent
            # data leakage.
            training_data = normalize_data(training_data, dataset)
            test_data = normalize_data(test_data, dataset)
            validation_data = normalize_data(validation_data, dataset)
            # Train model and evaluate.
            if type == "classification":
                neural_network_classifier_a = autoencoder_based_classifier(training_data, target_col, h1, h2)
                neural_network_classifier_a.train()
                predictions_a = neural_network_classifier_a.predict(test_data)
                performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
                
                neural_network_classifier_b = autoencoder_based_classifier(test_data, target_col, h1, h2)
                neural_network_classifier_b.train()
                predictions_b = neural_network_classifier_b.predict(training_data)
                performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            else:
                neural_network_regressor_a = autoencoder_based_regressor(training_data, target_col, h1, h2)
                neural_network_regressor_a.train()
                predictions_a = neural_network_regressor_a.predict(test_data)
                performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
                
                neural_network_regressor_b = autoencoder_based_regressor(test_data, target_col, h1, h2)
                neural_network_regressor_b.train()
                predictions_b = neural_network_regressor_b.predict(training_data)
                performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            average_performances.append(performance_a)
            average_performances.append(performance_b)
        average_performance = np.mean(average_performances)
        results_df.loc[index, eval_type] = average_performance
        print("h1 = " + str(h1))
        print("h2 = " + str(h2))
        print("Average error from 5x2 CV: " + str(average_performance))
        print("")
    best_err_ind = results_df[eval_type].idxmin()
    best_err = results_df.loc[best_err_ind, eval_type]
    best_h1 = results_df.loc[best_err_ind, "h1"]
    best_h2 = results_df.loc[best_err_ind, "h2"]
    print("Best Performance = " + str(best_err))
    print("Best h1 = " + str(best_h1))
    print("Best h2 = " + str(best_h2))
    print("")
    results_df.to_csv(output_file, index = False)

###################################################################################################
# Cancer data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = ["class_4"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment3/data/breast-cancer-wisconsin_autoencodernn_results.csv"

h1_list = [1, 5, 10, 25, 50]
h2_list = [1, 5, 10, 25, 50]

tune_hyperparameters(file_path, dataset, target_col, type, output_file, h1_list, h2_list)

###################################################################################################
# Car data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/car.data"
dataset = "car"
target_col = ["class_good", "class_unacc", "class_vgood", "class_acc"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment3/data/car_autoencodernn_results.csv"

h1_list = [1, 5, 10, 25, 50]
h2_list = [1, 5, 10, 25, 50]

tune_hyperparameters(file_path, dataset, target_col, type, output_file, h1_list, h2_list)

###################################################################################################
# Voting data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/house-votes-84.data"
dataset = "house-votes-84"
target_col = ["class_republican"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment3/data/house-votes-84_autoencodernn_results.csv"

h1_list = [1, 5, 10, 25, 50]
h2_list = [1, 5, 10, 25, 50]

tune_hyperparameters(file_path, dataset, target_col, type, output_file, h1_list, h2_list)

###################################################################################################
# Abalone data 
###################################################################################################

# Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/abalone.data"
dataset = "abalone"
target_col = ["rings"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment3/data/abalone_autoencodernn_results.csv"

h1_list = [1, 5, 10, 25, 50]
h2_list = [1, 5, 10, 25, 50]

tune_hyperparameters(file_path, dataset, target_col, type, output_file, h1_list, h2_list)

###################################################################################################
# Computer data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/machine.data"
dataset = "machine"
target_col = ["prp"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment3/data/machine_autoencodernn_results.csv"

h1_list = [1, 5, 10, 25, 50]
h2_list = [1, 5, 10, 25, 50]

tune_hyperparameters(file_path, dataset, target_col, type, output_file, h1_list, h2_list)

###################################################################################################
# Forest Fire data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/forestfires.csv"
dataset = "forestfires"
target_col = ["area"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment3/data/forestfires_autoencodernn_results.csv"

h1_list = [1, 5, 10, 25, 50]
h2_list = [1, 5, 10, 25, 50]

tune_hyperparameters(file_path, dataset, target_col, type, output_file, h1_list, h2_list)