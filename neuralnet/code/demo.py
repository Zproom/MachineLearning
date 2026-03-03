# demo.py
# This file contains code for the video demo.

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

###################################################################################################
# Cancer data 
###################################################################################################

# Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = ["class_4"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment3/data/breast-cancer-wisconsin_results.csv"

# Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation", random_state = 1)
training_data, test_data = stratify_classes(non_validation_data, target_col, random_state = 1)
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

# 1. Provide sample outputs from one test fold showing performance on each of your three networks.
# Train the logistic regression model and evaluate performance.
logistic_regression_model = logistic_regression(training_data, target_col)
logistic_regression_model.train()
predictions = logistic_regression_model.predict(test_data)
performance = evaluate("classerr", predictions, test_data, target_col)

# Compare to feedforward network.
neural_network_simple = neural_network_classifier(training_data, target_col, 5, 5)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("classerr", predictions_nn, test_data, target_col)

# Compare to autoencoder.
autoencoder_classifier = autoencoder_based_classifier(training_data, target_col, 5, 5)
autoencoder_classifier.train()
predictions_abnn = autoencoder_classifier.predict(test_data)
performance_abnn = evaluate("classerr", predictions_abnn, test_data, target_col)

# Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions_null = null_classifier.predict(len(test_data)).astype(int)
performance_null = evaluate("classerr", predictions_null, test_data, target_col)

print("Null Model Error: " + str(performance_null))
print("Logistic Regression Model Error: " + str(performance))
print("Regular Neural Network Error: " + str(performance_nn))
print("Autoencoder-based Network Error: " + str(performance_abnn))

# 2. Demonstrate and explain how an example is propagated through each network. Be sure to show
# the activations at each layer being calculated correctly.
predictions = logistic_regression_model.predict(test_data)
predictions_nn = neural_network_simple.predict(test_data)
predictions_abnn = autoencoder_classifier.predict(test_data)

# Reset for 3. 
logistic_regression_model = logistic_regression(training_data, target_col)

###################################################################################################
# Computer data 
###################################################################################################

# Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/machine.data"
dataset = "machine"
target_col = ["prp"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment3/data/machine_results.csv"

# Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation", random_state = 1)
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

# 3. Demonstrate the weight updates for the linear networks (one each).
logistic_regression_model.train()
# Train the model and evaluate performance.
linear_regression_model = linear_regression(training_data, target_col)
linear_regression_model.train()

# 4. Demonstrate the weight updates occurring on a two-layer network for each of the layers.
neural_network_simple = neural_network_regressor(training_data, target_col, 10, 10)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("mse", predictions_nn, test_data, target_col)

# 5. Demonstrate the weight updates occurring during the training of the autoencoder.
# 6. Demonstrate the autoencoding functionality (i.e., recovery of an input pattern on the output).
autoencoder_regressor = autoencoder_based_regressor(training_data, target_col, 10, 10)
autoencoder_regressor.train()
predictions_abnn = autoencoder_regressor.predict(test_data)
performance_abnn = evaluate("mse", predictions_abnn, test_data, target_col)