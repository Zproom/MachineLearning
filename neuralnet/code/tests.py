# tests.py
# This file contains tests to ensure the machine learning algorithm methods and helper functions 
# work properly.

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

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = ["class_4"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment3/data/breast-cancer-wisconsin_results.csv"

# 2. Load and prepare data.
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

# 3. Train the model and evaluate performance.
logistic_regression_model = logistic_regression(training_data, target_col)
logistic_regression_model.train()
predictions = logistic_regression_model.predict(test_data)
performance = evaluate("classerr", predictions, test_data, target_col)

# 4. Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions = null_classifier.predict(len(test_data)).astype(int)
performance = evaluate("classerr", predictions, test_data, target_col)
print("Null Model Error: " + str(performance))

# 5. Compare to feedforward network.
neural_network_simple = neural_network_classifier(training_data, target_col, 5, 5)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("classerr", predictions_nn, test_data, target_col)

# 6. Test autoencoder.
autoencoder_classifier = autoencoder_based_classifier(training_data, target_col, 5, 5)
autoencoder_classifier.train()
predictions_abnn = autoencoder_classifier.predict(test_data)
performance_abnn = evaluate("classerr", predictions_abnn, test_data, target_col)

###################################################################################################
# Car data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/car.data"
dataset = "car"
target_col = ["class_good", "class_unacc", "class_vgood", "class_acc"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment3/data/car_results.csv"

# 2. Load and prepare data.
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

# 3. Train the model and evaluate performance.
logistic_regression_model = logistic_regression(training_data, target_col)
logistic_regression_model.train()
predictions = logistic_regression_model.predict(test_data)
performance = evaluate("classerr", predictions, test_data, target_col)

# 4. Compare to null model.
training_labels = pd.Series(training_data[target_col].values.tolist()).reset_index(drop = True)
null_classifier = null_model(training_labels, type)
predictions = null_classifier.predict(len(test_data))
performance = evaluate("classerr", predictions, test_data, target_col)
print("Null Model Error: " + str(performance))

# 5. Compare to feedforward network.
neural_network_simple = neural_network_classifier(training_data, target_col, 5, 5)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("classerr", predictions_nn, test_data, target_col)

# 6. Test autoencoder.
autoencoder_classifier = autoencoder_based_classifier(training_data, target_col, 5, 5)
autoencoder_classifier.train()
predictions_abnn = autoencoder_classifier.predict(test_data)
performance_abnn = evaluate("classerr", predictions_abnn, test_data, target_col)

###################################################################################################
# Voting data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/house-votes-84.data"
dataset = "house-votes-84"
target_col = ["class_republican"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment3/data/house-votes-84_results.csv"

# 2. Load and prepare data.
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

# 3. Train the model and evaluate performance.
logistic_regression_model = logistic_regression(training_data, target_col)
logistic_regression_model.train()
predictions = logistic_regression_model.predict(test_data)
performance = evaluate("classerr", predictions, test_data, target_col)

# 4. Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions = null_classifier.predict(len(test_data))
performance = evaluate("classerr", predictions, test_data, target_col)
print("Null Model Error: " + str(performance))

# 5. Compare to feedforward network.
neural_network_simple = neural_network_classifier(training_data, target_col, 5, 5)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("classerr", predictions_nn, test_data, target_col)

# 6. Test autoencoder.
autoencoder_classifier = autoencoder_based_classifier(training_data, target_col, 5, 5)
autoencoder_classifier.train()
predictions_abnn = autoencoder_classifier.predict(test_data)
performance_abnn = evaluate("classerr", predictions_abnn, test_data, target_col)

###################################################################################################
# Abalone data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/abalone.data"
dataset = "abalone"
target_col = ["rings"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment3/data/abalone_results.csv"

# 2. Load and prepare data.
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

# 3. Train the model and evaluate performance.
linear_regression_model = linear_regression(training_data, target_col)
linear_regression_model.train()
predictions = linear_regression_model.predict(test_data)
performance = evaluate("mse", predictions, test_data, target_col)

# 4. Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions = null_classifier.predict(len(test_data))
performance = evaluate("mse", predictions, test_data, target_col)
print("Null Model Error: " + str(performance))

# 5. Compare to feedforward network.
neural_network_simple = neural_network_regressor(training_data, target_col, 5, 5)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("mse", predictions_nn, test_data, target_col)

# 6. Test autoencoder.
autoencoder_regressor = autoencoder_based_regressor(training_data, target_col, 5, 5)
autoencoder_regressor.train()
predictions_abnn = autoencoder_regressor.predict(test_data)
performance_abnn = evaluate("mse", predictions_abnn, test_data, target_col)

###################################################################################################
# Computer data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment3/data/machine.data"
dataset = "machine"
target_col = ["prp"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment3/data/machine_results.csv"

# 2. Load and prepare data.
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

# 3. Train the model and evaluate performance.
linear_regression_model = linear_regression(training_data, target_col)
linear_regression_model.train()
predictions = linear_regression_model.predict(test_data)
performance = evaluate("mse", predictions, test_data, target_col)

# 4. Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions = null_classifier.predict(len(test_data))
performance = evaluate("mse", predictions, test_data, target_col)
print("Null Model Error: " + str(performance))

# 5. Compare to feedforward network.
neural_network_simple = neural_network_regressor(training_data, target_col, 10, 10)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("mse", predictions_nn, test_data, target_col)

# 6. Test autoencoder.
autoencoder_regressor = autoencoder_based_regressor(training_data, target_col, 10, 10)
autoencoder_regressor.train()
predictions_abnn = autoencoder_regressor.predict(test_data)
performance_abnn = evaluate("mse", predictions_abnn, test_data, target_col)

###################################################################################################
# Forest Fire data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/forestfires.csv"
dataset = "forestfires"
target_col = ["area"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment1/data/forestfires_results.csv"

# 2. Load and prepare data.
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

# 3. Train the model and evaluate performance.
linear_regression_model = linear_regression(training_data, target_col)
linear_regression_model.train()
predictions = linear_regression_model.predict(test_data)
performance = evaluate("mse", predictions, test_data, target_col)

# 4. Compare to null model.
null_classifier = null_model(training_data[target_col], type)
predictions = null_classifier.predict(len(test_data))
performance = evaluate("mse", predictions, test_data, target_col)
print("Null Model Error: " + str(performance))

# 5. Compare to feedforward network.
neural_network_simple = neural_network_regressor(training_data, target_col, 10, 10)
neural_network_simple.train()
predictions_nn = neural_network_simple.predict(test_data)
performance_nn = evaluate("mse", predictions_nn, test_data, target_col)

# 6. Test autoencoder.
autoencoder_regressor = autoencoder_based_regressor(training_data, target_col, 10, 10)
autoencoder_regressor.train()
predictions_abnn = autoencoder_regressor.predict(test_data)
performance_abnn = evaluate("mse", predictions_abnn, test_data, target_col)