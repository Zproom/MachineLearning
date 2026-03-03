# demo.py
# This file is used to demonstrate the proper functioning of the code for the video.

from regular_knn import *
from edited_knn import *
from null_model import *
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *

# Use computer hardware data.

# Specify parameters.
# Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/machine.data"
dataset = "machine"
target_col = ["prp"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment1/data/machine_results.csv"

input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)

# 1. Construction of separate subsets of the data to support 5 × 2 cross validation (i.e., the held-out
# 20% and then two folds of 40% each).
validation_data, non_validation_data = random_split(input_data, "validation")
print(input_data)
print(validation_data)
print(non_validation_data)
training_data, test_data = random_split(non_validation_data, "testing")
print(training_data)
print(test_data)

# 2. The calculation of your distance function(s).

# 3. The calculation of your kernel function.

# 4. An example of a point being regressed using k-nn. Show the neighbors returned as well as the
# point being classified.
k = 3
bandwidth = 1
classifier = knn_regressor(k, bandwidth, training_data)
predictions = classifier.predict(validation_data, target_col)

# 5. An example of a point being regressed using your null regression model.
regressor_null = null_model(training_data[target_col], "regression")
predictions_null = regressor_null.predict(len(validation_data))
print(predictions_null)

# 6. An example of a point being classified using your null classification model.
# Use breast cancer data.

# 7. Numerical data normalization through min-max normalization.

# 8. Categorical data transformed either to be one-hot coded or to incorporate some kind of label
# (integer) coding.

# Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = ["class_4"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin_results.csv"

# Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")

training_data, test_data = stratify_classes(non_validation_data, target_col)
    
classifier = null_model(training_data[target_col], "classification")
predictions = classifier.predict(len(validation_data))
print(predictions)

# 9. An example of a point being classified using k-nn. Show the neighbors returned as well as the
# point being classified.
k = 3
classifier = knn_classifier(k, training_data)
predictions = classifier.predict(validation_data, target_col)

# 10. An example being edited from the training set if you implemented edited nearest neighbor, or an
# example being added to the training set if you implemented condensed nearest neighbor.
# Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/forestfires.csv"
dataset = "forestfires"
target_col = ["area"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment1/data/forestfires_results_edited.csv"

# Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")

# Tune k. Perform 5x2 cross-validation for each group of parameter settings. Average the 
# results of the ten experiments for each group of parameter settings and pick the parameter
# settings with the best performance.
k_values = [5, 10, 50]
bandwidth_values = [0.1, 10]
threshold_values = [1, 5, 50]
all_parameter_settings = [(k, b, t) for k in k_values for b in bandwidth_values for t in 
                          threshold_values]
results_df = pd.DataFrame(all_parameter_settings, columns = ["k", "bandwidth", "threshold"])
results_df["mse"] = np.nan
for index, row in results_df.iterrows():
    k = results_df.loc[index, "k"]
    b = results_df.loc[index, "bandwidth"]
    t = results_df.loc[index, "threshold"]
    average_performances = []
    for i in range(5):
        training_data_a, training_data_b = random_split(non_validation_data, "testing")
        
        regressor_a = edited_knn_regressor(k, b, t, training_data_a)
        regressor_a.train(validation_data, target_col)
        predictions_a = regressor_a.predict(validation_data, target_col)
        performance_a = evaluate("mse", predictions_a, validation_data, target_col)
        
        regressor_b = edited_knn_regressor(k, b, t, training_data_b)
        regressor_b.train(validation_data, target_col)
        predictions_b = regressor_b.predict(validation_data, target_col)
        performance_b = evaluate("mse", predictions_b, validation_data, target_col)

        average_performances.append(performance_a)
        average_performances.append(performance_b)
    average_performance = np.mean(average_performances)
    results_df.loc[index, "mse"] = average_performance
    print("k = " + str(k))
    print("b = " + str(b))
    print("t = " + str(t))
    print("Average error from 5x2 CV: " + str(average_performance))
    print("") 
