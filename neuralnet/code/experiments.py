# experiments.py
# This file contains the experiments where I test the performance of the null model, linear 
# regression model, logistic regression model, classification and regression neural networks,
# and autoencoder-based classification and regression neural networks. The hidden node parameters 
# are obtained from running the hyperparameter_tuning*.py files.

import os
from null_model import *
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *
from logistic_regression import *
from linear_regression import *
from neural_network_classifier import *
from neural_network_regressor import *
from autoencoder_based_classifier import *
from autoencoder_based_regressor import *

def run_experiment(file_path, dataset, target_col, type, h1_regnn_class_size = None, 
                   h2_regnn_class_size = None, h1_regnn_reg_size = None, h2_regnn_reg_size = None,
                   h1_autoenc_class_size = None, h2_autoenc_class_size = None,
                   h1_autoenc_reg_size = None, h2_autoenc_reg_size = None):
    """This function runs a full experiment for a UCI dataset. It uses 5 x 2 cross-validation for 
    each model: (1) null model, (2) linear regression model, (3) logistic regression model, 
    (4) classification and regression neural networks, (5) and autoencoder-based classification and 
    regression neural networks.

    Args:
        file_path (str): The full file path to the UCI data file.
        dataset (str): The name of the UCI dataset. 
        target_col (str): The name of the target column.
        type (str): The type of experiment. It can be either "classification" or "regression".
        output_file (str): The file to write output to for analysis. This stores data on the number
        of nodes vs. the error during the pruning process.
        h1_regnn_class_size (int, optional): The number of nodes in the first hidden layer of a 
        regular neural network (classification). Defaults to None.
        h2_regnn_class_size (int, optional): The number of nodes in the second hidden layer of a 
        regular neural network (classification). Defaults to None.
        h1_regnn_reg_size (int, optional): The number of nodes in the first hidden layer of a 
        regular neural network (regression). Defaults to None.
        h2_regnn_reg_size (int, optional): The number of nodes in the second hidden layer of a 
        regular neural network (regression). Defaults to None.
        h1_autoenc_class_size (int, optional): The number of nodes in the first hidden layer of an 
        autoencoder neural network (classification). Defaults to None.
        h2_autoenc_class_size (int, optional): The number of nodes in the second hidden layer of an 
        autoencoder neural network (classification). Defaults to None.
        h1_autoenc_reg_size (int, optional): The number of nodes in the first hidden layer of an 
        autoencoder neural network (regression). Defaults to None.
        h2_autoenc_reg_size (int, optional): The number of nodes in the second hidden layer of an 
        autoencoder neural network (regression). Defaults to None.

    Raises:
        nothing. Prints the average result of the ten experiments for each model.
    """
    if type != "classification" and type != "regression":
        raise Exception("type must be 'classification' or 'regression'.")
    eval_type = "classerr" if type == "classification" else "mse"
    print("Running experiments for the following dataset: " + dataset)
    performances_null_model = []
    if type == "classification":
        performances_logistic_reg = []
        performances_neural_classifier = []
        performances_autoencoder_classifier = []
        for i in range(5):
            # Data prep
            input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
            validation_data, non_validation_data = random_split(input_data, "validation", 
                                                                random_state = i)
            training_data, test_data = stratify_classes(non_validation_data, target_col, 
                                                        random_state = i)
            # Handle missing values. Use the imputed training value (mean of non-missing 
            # observations) to impute missing test and validation data.
            training_data, imputed_val = handle_missing_values(training_data, dataset)
            test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
            validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]
            # Normalize the data. Normalize the training, test, and validation separately to 
            # prevent data leakage.
            training_data = normalize_data(training_data, dataset)
            test_data = normalize_data(test_data, dataset)
            validation_data = normalize_data(validation_data, dataset)
            # Train models and evaluate.
            # Null model
            if dataset == "car":
                training_labels = \
                    pd.Series(training_data[target_col].values.tolist()).reset_index(drop = True)
                null_classifier_a = null_model(training_labels, type)
                predictions_a = null_classifier_a.predict(len(test_data))
                performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
                training_labels = \
                    pd.Series(test_data[target_col].values.tolist()).reset_index(drop = True)
                null_classifier_b = null_model(training_labels, type)
                predictions_b = null_classifier_b.predict(len(training_data))
                performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            else:
                null_classifier_a = null_model(training_data[target_col], type)
                predictions_a = null_classifier_a.predict(len(test_data)).astype(int)
                performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
                null_classifier_b = null_model(test_data[target_col], type)
                predictions_b = null_classifier_a.predict(len(training_data)).astype(int)
                performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_null_model.extend([performance_a, performance_b])
            # Logistic regression
            logistic_regression_model_a = logistic_regression(training_data, target_col)
            logistic_regression_model_a.train()
            predictions_a = logistic_regression_model_a.predict(test_data)
            performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
            logistic_regression_model_b = logistic_regression(test_data, target_col)
            logistic_regression_model_b.train()
            predictions_b = logistic_regression_model_b.predict(training_data)
            performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_logistic_reg.extend([performance_a, performance_b])
            # Regular neural classifier
            neural_network_classifier_a = neural_network_classifier(training_data, target_col, 
                                                                    h1_regnn_class_size, 
                                                                    h2_regnn_class_size)
            neural_network_classifier_a.train()
            predictions_a = neural_network_classifier_a.predict(test_data)
            performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
            neural_network_classifier_b = neural_network_classifier(test_data, target_col, 
                                                                    h1_regnn_class_size, 
                                                                    h2_regnn_class_size)
            neural_network_classifier_b.train()
            predictions_b = neural_network_classifier_b.predict(training_data)
            performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_neural_classifier.extend([performance_a, performance_b])
            # Autoencoder-based neural classifier
            autoencoder_classifier_a = autoencoder_based_classifier(training_data, target_col, 
                                                                    h1_autoenc_class_size, 
                                                                    h2_autoenc_class_size)
            autoencoder_classifier_a.train()
            predictions_a = autoencoder_classifier_a.predict(test_data)
            performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
            autoencoder_classifier_b = autoencoder_based_classifier(test_data, target_col, 
                                                                    h1_autoenc_class_size, 
                                                                    h2_autoenc_class_size)
            autoencoder_classifier_b.train()
            predictions_b = autoencoder_classifier_b.predict(training_data)
            performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_autoencoder_classifier.extend([performance_a, performance_b])
        average_performance_null_model = np.mean(performances_null_model)
        average_performance_logistic_reg = np.mean(performances_logistic_reg)
        average_performance_neural_classifier = np.mean(performances_neural_classifier)
        average_performance_autoencoder_classifier = np.mean(performances_autoencoder_classifier)
        print("Null Model Error (5x2 CV Avg): " + str(average_performance_null_model))
        print("Logistic Regression Error (5x2 CV Avg): " + str(average_performance_logistic_reg))
        print("Neural Classifier Error (5x2 CV Avg): " + 
              str(average_performance_neural_classifier))
        print("Autoencoder Classifier Error (5x2 CV Avg): " + 
              str(average_performance_autoencoder_classifier))
        print("")
    else:
        performances_linear_reg = []
        performances_neural_reg = []
        performances_autoencoder_reg = []
        for i in range(5):
            # Data prep
            input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
            validation_data, non_validation_data = random_split(input_data, "validation", 
                                                                random_state = i)
            training_data, test_data = random_split(non_validation_data, "testing", 
                                                    random_state = 1)
            # Handle missing values. Use the imputed training value (mean of non-missing 
            # observations) to impute missing test and validation data.
            training_data, imputed_val = handle_missing_values(training_data, dataset)
            test_data = handle_missing_values(test_data, dataset, imputed_val)[0]
            validation_data = handle_missing_values(validation_data, dataset, imputed_val)[0]
            # Normalize the data. Normalize the training, test, and validation separately to 
            # prevent data leakage.
            training_data = normalize_data(training_data, dataset)
            test_data = normalize_data(test_data, dataset)
            validation_data = normalize_data(validation_data, dataset)
            # Train models and evaluate.
            # Null model
            null_classifier_a = null_model(training_data[target_col], type)
            predictions_a = null_classifier_a.predict(len(test_data)).astype(int)
            performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
            null_classifier_b = null_model(test_data[target_col], type)
            predictions_b = null_classifier_a.predict(len(training_data)).astype(int)
            performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_null_model.extend([performance_a, performance_b])
            # Linear regression
            linear_regression_model_a = linear_regression(training_data, target_col)
            linear_regression_model_a.train()
            predictions_a = linear_regression_model_a.predict(test_data)
            performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
            linear_regression_model_b = linear_regression(test_data, target_col)
            linear_regression_model_b.train()
            predictions_b = linear_regression_model_b.predict(training_data)
            performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_linear_reg.extend([performance_a, performance_b])
            # Regular neural regression
            neural_network_regressor_a = neural_network_regressor(training_data, target_col, 
                                                                  h1_regnn_reg_size, 
                                                                  h2_regnn_reg_size)
            neural_network_regressor_a.train()
            predictions_a = neural_network_regressor_a.predict(test_data)
            performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
            neural_network_regressor_b = neural_network_regressor(test_data, target_col, 
                                                                  h1_regnn_reg_size, 
                                                                  h2_regnn_reg_size)
            neural_network_regressor_b.train()
            predictions_b = neural_network_regressor_b.predict(training_data)
            performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_neural_reg.extend([performance_a, performance_b])
            # Autoencoder-based neural regressor
            autoencoder_regressor_a = autoencoder_based_regressor(training_data, target_col, 
                                                                  h1_autoenc_reg_size, 
                                                                  h2_autoenc_reg_size)
            autoencoder_regressor_a.train()
            predictions_a = autoencoder_regressor_a.predict(test_data)
            performance_a = evaluate(eval_type, predictions_a, test_data, target_col)
            autoencoder_regressor_b = autoencoder_based_regressor(test_data, target_col, 
                                                                  h1_autoenc_reg_size, 
                                                                  h2_autoenc_reg_size)
            autoencoder_regressor_b.train()
            predictions_b = autoencoder_regressor_b.predict(training_data)
            performance_b = evaluate(eval_type, predictions_b, training_data, target_col)
            performances_autoencoder_reg.extend([performance_a, performance_b])
        average_performance_null_model = np.mean(performances_null_model)
        average_performance_linear_reg = np.mean(performances_linear_reg)
        average_performance_neural_reg = np.mean(performances_neural_reg)
        average_performance_autoencoder_reg = np.mean(performances_autoencoder_reg)
        print("Null Model Error (5x2 CV Avg): " + str(average_performance_null_model))
        print("Linear Regression Error (5x2 CV Avg): " + str(average_performance_linear_reg))
        print("Neural Regression Error (5x2 CV Avg): " + 
              str(average_performance_neural_reg))
        print("Autoencoder Regression Error (5x2 CV Avg): " + 
              str(average_performance_autoencoder_reg))
        print("")

###################################################################################################
# Cancer data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = ["class_4"]
type = "classification"

run_experiment(file_path, dataset, target_col, type, 
               h1_regnn_class_size = 1,
               h2_regnn_class_size = 50,
               h1_autoenc_class_size = 10,
               h2_autoenc_class_size = 25)

###################################################################################################
# Car data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/car.data"
dataset = "car"
target_col = ["class_good", "class_unacc", "class_vgood", "class_acc"]
type = "classification"

run_experiment(file_path, dataset, target_col, type, 
               h1_regnn_class_size = 5,
               h2_regnn_class_size = 50,
               h1_autoenc_class_size = 5,
               h2_autoenc_class_size = 5)

###################################################################################################
# Voting data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/house-votes-84.data"
dataset = "house-votes-84"
target_col = ["class_republican"]
type = "classification"

run_experiment(file_path, dataset, target_col, type, 
               h1_regnn_class_size = 5,
               h2_regnn_class_size = 5,
               h1_autoenc_class_size = 25,
               h2_autoenc_class_size = 25)

###################################################################################################
# Abalone data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/abalone.data"
dataset = "abalone"
target_col = ["rings"]
type = "regression"

run_experiment(file_path, dataset, target_col, type, 
               h1_regnn_reg_size = 1,
               h2_regnn_reg_size = 5,
               h1_autoenc_reg_size = 50,
               h2_autoenc_reg_size = 50)

###################################################################################################
# Computer data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment3/data/machine.data"
dataset = "machine"
target_col = ["prp"]
type = "regression"

run_experiment(file_path, dataset, target_col, type, 
               h1_regnn_reg_size = 50,
               h2_regnn_reg_size = 25,
               h1_autoenc_reg_size = 10,
               h2_autoenc_reg_size = 50)

###################################################################################################
# Forest Fire data 
###################################################################################################

file_path = "/home/zacharyproom/IntroML/assignment1/data/forestfires.csv"
dataset = "forestfires"
target_col = ["area"]
type = "regression"

run_experiment(file_path, dataset, target_col, type, 
               h1_regnn_reg_size = 5,
               h2_regnn_reg_size = 1,
               h1_autoenc_reg_size = 25,
               h2_autoenc_reg_size = 10)