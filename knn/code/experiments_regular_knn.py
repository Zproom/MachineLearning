# experiments_regular_knn.py
# This file contains the experiments where fine tuning is done on parameters. The experiments test
# regular (i.e. non-edited) k-nearest neighbors classification and regression.

from regular_knn import *
from edited_knn import *
from null_model import *
from data_processing_functions import *
from evaluation_functions import *
from cross_validation_functions import *

###################################################################################################
# Cancer data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin.data"
dataset = "breast-cancer-wisconsin"
target_col = ["class_4"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment1/data/breast-cancer-wisconsin_results.csv"

# 2. Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")

# 3. Tune k. Perform 5x2 cross-validation for each group of parameter settings. Average the 
# results of the ten experiments for each group of parameter settings and pick the parameter
# settings with the best performance.
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
average_performances_k = []
for k in k_values:
    average_performances = []
    for i in range(5):
        training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
        
        classifier_a = knn_classifier(k, training_data_a)
        predictions_a = classifier_a.predict(validation_data, target_col)
        performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
        
        classifier_b = knn_classifier(k, training_data_b)
        predictions_b = classifier_b.predict(validation_data, target_col)
        performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

        average_performances.append(performance_a)
        average_performances.append(performance_b)
    average_performance = np.mean(average_performances)
    average_performances_k.append(average_performance)
    print("k = " + str(k))
    print("Average error from 5x2 CV: " + str(average_performance))
    print("")    

# 4. Perform 5x2 cross-validation using the best parameter settings. Average the results of the ten
# experiments and report the average.
best_k = k_values[average_performances_k.index(min(average_performances_k))]
print("Best k = " + str(best_k))
average_performances = []
for i in range(5):
    training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
    
    classifier_a = knn_classifier(best_k, training_data_a)
    predictions_a = classifier_a.predict(validation_data, target_col)
    performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
    
    classifier_b = knn_classifier(best_k, training_data_b)
    predictions_b = classifier_b.predict(validation_data, target_col)
    performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("k = " + str(best_k))
print("Average error from 5x2 CV with best k: " + str(average_performance))
print("")
output_data = pd.DataFrame({"k": k_values, "error": average_performances_k})
output_data.to_csv(output_file, index = False)

# 5. Perform 5x2 cross-validation using the null model. Average the results of the ten experiments
# and report the average.
average_performances = []
for i in range(5):
    training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
    
    classifier_a = null_model(training_data_a[target_col], "classification")
    predictions_a = classifier_a.predict(len(validation_data))
    performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
    
    classifier_b = null_model(training_data_b[target_col], "classification")
    predictions_b = classifier_b.predict(len(validation_data))
    performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("Average error from 5x2 CV with null model: " + str(average_performance))
print("")

###################################################################################################
# Car data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/car.data"
dataset = "car"
target_col = ["class_good", "class_unacc", "class_vgood"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment1/data/car_results.csv"

# 2. Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")

# 3. Tune k. Perform 5x2 cross-validation for each group of parameter settings. Average the 
# results of the ten experiments for each group of parameter settings and pick the parameter
# settings with the best performance.
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
average_performances_k = []
for k in k_values:
    average_performances = []
    for i in range(5):
        training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
        
        classifier_a = knn_classifier(k, training_data_a)
        predictions_a = classifier_a.predict(validation_data, target_col)
        performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
        
        classifier_b = knn_classifier(k, training_data_b)
        predictions_b = classifier_b.predict(validation_data, target_col)
        performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

        average_performances.append(performance_a)
        average_performances.append(performance_b)
    average_performance = np.mean(average_performances)
    average_performances_k.append(average_performance)
    print("k = " + str(k))
    print("Average error from 5x2 CV: " + str(average_performance))
    print("")    

# 4. Perform 5x2 cross-validation using the best parameter settings. Average the results of the ten
# experiments and report the average.
best_k = k_values[average_performances_k.index(min(average_performances_k))]
print("Best k = " + str(best_k))
average_performances = []
for i in range(5):
    training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
    
    classifier_a = knn_classifier(best_k, training_data_a)
    predictions_a = classifier_a.predict(validation_data, target_col)
    performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
    
    classifier_b = knn_classifier(best_k, training_data_b)
    predictions_b = classifier_b.predict(validation_data, target_col)
    performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("k = " + str(best_k))
print("Average error from 5x2 CV with best k: " + str(average_performance))
print("")
output_data = pd.DataFrame({"k": k_values, "error": average_performances_k})
output_data.to_csv(output_file, index = False)

# 5. Perform 5x2 cross-validation using the null model. Average the results of the ten experiments
# and report the average.
average_performances = []
for i in range(5):
    training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
    
    training_labels = pd.Series(training_data_a[target_col].values.tolist()).reset_index(drop = True)
    classifier_a = null_model(training_labels, "classification")
    predictions_a = classifier_a.predict(len(validation_data))
    performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
    
    training_labels = pd.Series(training_data_b[target_col].values.tolist()).reset_index(drop = True)
    classifier_b = null_model(training_labels, "classification")
    predictions_b = classifier_b.predict(len(validation_data))
    performance_b = evaluate("classerr", predictions_b, validation_data, target_col)
    
    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("Average error from 5x2 CV with null model: " + str(average_performance))
print("")

###################################################################################################
# Voting data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/house-votes-84.data"
dataset = "house-votes-84"
target_col = ["class_republican"]
type = "classification"
output_file = "/home/zacharyproom/IntroML/assignment1/data/house-votes-84_results.csv"

# 2. Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")

# 3. Tune k. Perform 5x2 cross-validation for each group of parameter settings. Average the 
# results of the ten experiments for each group of parameter settings and pick the parameter
# settings with the best performance.
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 30, 40, 50]
average_performances_k = []
for k in k_values:
    average_performances = []
    for i in range(5):
        training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
        
        classifier_a = knn_classifier(k, training_data_a)
        predictions_a = classifier_a.predict(validation_data, target_col)
        performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
        
        classifier_b = knn_classifier(k, training_data_b)
        predictions_b = classifier_b.predict(validation_data, target_col)
        performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

        average_performances.append(performance_a)
        average_performances.append(performance_b)
    average_performance = np.mean(average_performances)
    average_performances_k.append(average_performance)
    print("k = " + str(k))
    print("Average error from 5x2 CV: " + str(average_performance))
    print("")    

# 4. Perform 5x2 cross-validation using the best parameter settings. Average the results of the ten
# experiments and report the average.
best_k = k_values[average_performances_k.index(min(average_performances_k))]
print("Best k = " + str(best_k))
average_performances = []
for i in range(5):
    training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
    
    classifier_a = knn_classifier(best_k, training_data_a)
    predictions_a = classifier_a.predict(validation_data, target_col)
    performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
    
    classifier_b = knn_classifier(best_k, training_data_b)
    predictions_b = classifier_b.predict(validation_data, target_col)
    performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("k = " + str(best_k))
print("Average error from 5x2 CV with best k: " + str(average_performance))
print("")
output_data = pd.DataFrame({"k": k_values, "error": average_performances_k})
output_data.to_csv(output_file, index = False)

# 5. Perform 5x2 cross-validation using the null model. Average the results of the ten experiments
# and report the average.
average_performances = []
for i in range(5):
    training_data_a, training_data_b = stratify_classes(non_validation_data, target_col)
    
    classifier_a = null_model(training_data_a[target_col], "classification")
    predictions_a = classifier_a.predict(len(validation_data))
    performance_a = evaluate("classerr", predictions_a, validation_data, target_col)
    
    classifier_b = null_model(training_data_b[target_col], "classification")
    predictions_b = classifier_b.predict(len(validation_data))
    performance_b = evaluate("classerr", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("Average error from 5x2 CV with null model: " + str(average_performance))
print("")

###################################################################################################
# Abalone data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/abalone.data"
dataset = "abalone"
target_col = ["rings"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment1/data/abalone_results.csv"

# 2. Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")

# 3. Tune k. Perform 5x2 cross-validation for each group of parameter settings. Average the 
# results of the ten experiments for each group of parameter settings and pick the parameter
# settings with the best performance.
k_values = [1, 5, 10, 20, 30, 40, 50]
bandwidth_values = [0.1, 0.5, 1, 10]
all_parameter_settings = [(k, b) for k in k_values for b in bandwidth_values]
results_df = pd.DataFrame(all_parameter_settings, columns = ["k", "bandwidth"])
results_df["mse"] = np.nan
for index, row in results_df.iterrows():
    k = results_df.loc[index, "k"]
    b = results_df.loc[index, "bandwidth"]
    average_performances = []
    for i in range(5):
        training_data_a, training_data_b = random_split(non_validation_data, "testing")
        
        regressor_a = knn_regressor(k, b, training_data_a)
        predictions_a = regressor_a.predict(validation_data, target_col)
        performance_a = evaluate("mse", predictions_a, validation_data, target_col)
        
        regressor_b = knn_regressor(k, b, training_data_b)
        predictions_b = regressor_b.predict(validation_data, target_col)
        performance_b = evaluate("mse", predictions_b, validation_data, target_col)

        average_performances.append(performance_a)
        average_performances.append(performance_b)
    average_performance = np.mean(average_performances)
    results_df.loc[index, "mse"] = average_performance
    print("k = " + str(k))
    print("b = " + str(b))
    print("Average error from 5x2 CV: " + str(average_performance))
    print("")    

# 4. Perform 5x2 cross-validation using the best parameter settings. Average the results of the ten
# experiments and report the average.
best_mse_ind = results_df["mse"].idxmin()
best_k = results_df.loc[best_mse_ind, "k"]
best_bandwidth = results_df.loc[best_mse_ind, "bandwidth"]
print("Best k = " + str(best_k))
print("Best bandwidth = " + str(best_bandwidth))
k = best_k
b = best_bandwidth
average_performances = []
for i in range(5):
    training_data_a, training_data_b = random_split(non_validation_data, "testing")
        
    regressor_a = knn_regressor(k, b, training_data_a)
    predictions_a = regressor_a.predict(validation_data, target_col)
    performance_a = evaluate("mse", predictions_a, validation_data, target_col)
    
    regressor_b = knn_regressor(k, b, training_data_b)
    predictions_b = regressor_b.predict(validation_data, target_col)
    performance_b = evaluate("mse", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("k = " + str(k))
print("bandwidth = " + str(b))
print("Average error from 5x2 CV with best k: " + str(average_performance))
print("")
results_df.to_csv(output_file, index = False)

# 5. Perform 5x2 cross-validation using the null model. Average the results of the ten experiments
# and report the average.
average_performances = []
for i in range(5):
    training_data_a, training_data_b = random_split(non_validation_data, "testing")
    
    regressor_a = null_model(training_data_a[target_col], "regression")
    predictions_a = regressor_a.predict(len(validation_data))
    performance_a = evaluate("mse", predictions_a, validation_data, target_col)
    
    regressor_b = null_model(training_data_b[target_col], "regression")
    predictions_b = regressor_b.predict(len(validation_data))
    performance_b = evaluate("mse", predictions_b, validation_data, target_col)
    
    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("Average error from 5x2 CV with null model: " + str(average_performance))
print("")

###################################################################################################
# Computer data 
###################################################################################################

# 1. Specify parameters.
file_path = "/home/zacharyproom/IntroML/assignment1/data/machine.data"
dataset = "machine"
target_col = ["prp"]
type = "regression"
output_file = "/home/zacharyproom/IntroML/assignment1/data/machine_results.csv"

# 2. Load and prepare data.
input_data, excluded_cols = run_data_loading_pipeline(file_path, dataset)
validation_data, non_validation_data = random_split(input_data, "validation")

# 3. Tune k. Perform 5x2 cross-validation for each group of parameter settings. Average the 
# results of the ten experiments for each group of parameter settings and pick the parameter
# settings with the best performance.
k_values = [1, 5, 10, 20, 30, 40, 50]
bandwidth_values = [0.1, 0.5, 1, 10]
all_parameter_settings = [(k, b) for k in k_values for b in bandwidth_values]
results_df = pd.DataFrame(all_parameter_settings, columns = ["k", "bandwidth"])
results_df["mse"] = np.nan
for index, row in results_df.iterrows():
    k = results_df.loc[index, "k"]
    b = results_df.loc[index, "bandwidth"]
    average_performances = []
    for i in range(5):
        training_data_a, training_data_b = random_split(non_validation_data, "testing")
        
        regressor_a = knn_regressor(k, b, training_data_a)
        predictions_a = regressor_a.predict(validation_data, target_col)
        performance_a = evaluate("mse", predictions_a, validation_data, target_col)
        
        regressor_b = knn_regressor(k, b, training_data_b)
        predictions_b = regressor_b.predict(validation_data, target_col)
        performance_b = evaluate("mse", predictions_b, validation_data, target_col)

        average_performances.append(performance_a)
        average_performances.append(performance_b)
    average_performance = np.mean(average_performances)
    results_df.loc[index, "mse"] = average_performance
    print("k = " + str(k))
    print("b = " + str(b))
    print("Average error from 5x2 CV: " + str(average_performance))
    print("")    

# 4. Perform 5x2 cross-validation using the best parameter settings. Average the results of the ten
# experiments and report the average.
best_mse_ind = results_df["mse"].idxmin()
best_k = results_df.loc[best_mse_ind, "k"]
best_bandwidth = results_df.loc[best_mse_ind, "bandwidth"]
print("Best k = " + str(best_k))
print("Best bandwidth = " + str(best_bandwidth))
k = best_k
b = best_bandwidth
average_performances = []
for i in range(5):
    training_data_a, training_data_b = random_split(non_validation_data, "testing")
        
    regressor_a = knn_regressor(k, b, training_data_a)
    predictions_a = regressor_a.predict(validation_data, target_col)
    performance_a = evaluate("mse", predictions_a, validation_data, target_col)
    
    regressor_b = knn_regressor(k, b, training_data_b)
    predictions_b = regressor_b.predict(validation_data, target_col)
    performance_b = evaluate("mse", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("k = " + str(k))
print("bandwidth = " + str(b))
print("Average error from 5x2 CV with best k: " + str(average_performance))
print("")
results_df.to_csv(output_file, index = False)

# 5. Perform 5x2 cross-validation using the null model. Average the results of the ten experiments
# and report the average.
average_performances = []
for i in range(5):
    training_data_a, training_data_b = random_split(non_validation_data, "testing")
    
    regressor_a = null_model(training_data_a[target_col], "regression")
    predictions_a = regressor_a.predict(len(validation_data))
    performance_a = evaluate("mse", predictions_a, validation_data, target_col)
    
    regressor_b = null_model(training_data_b[target_col], "regression")
    predictions_b = regressor_b.predict(len(validation_data))
    performance_b = evaluate("mse", predictions_b, validation_data, target_col)
    
    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("Average error from 5x2 CV with null model: " + str(average_performance))
print("")

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
validation_data, non_validation_data = random_split(input_data, "validation")

# 3. Tune k. Perform 5x2 cross-validation for each group of parameter settings. Average the 
# results of the ten experiments for each group of parameter settings and pick the parameter
# settings with the best performance.
k_values = [1, 5, 10, 20, 30, 40, 50]
bandwidth_values = [0.1, 0.5, 1, 10]
all_parameter_settings = [(k, b) for k in k_values for b in bandwidth_values]
results_df = pd.DataFrame(all_parameter_settings, columns = ["k", "bandwidth"])
results_df["mse"] = np.nan
for index, row in results_df.iterrows():
    k = results_df.loc[index, "k"]
    b = results_df.loc[index, "bandwidth"]
    average_performances = []
    for i in range(5):
        training_data_a, training_data_b = random_split(non_validation_data, "testing")
        
        regressor_a = knn_regressor(k, b, training_data_a, True)
        predictions_a = regressor_a.predict(validation_data, target_col)
        performance_a = evaluate("mse", predictions_a, validation_data, target_col)
        
        regressor_b = knn_regressor(k, b, training_data_b, True)
        predictions_b = regressor_b.predict(validation_data, target_col)
        performance_b = evaluate("mse", predictions_b, validation_data, target_col)

        average_performances.append(performance_a)
        average_performances.append(performance_b)
    average_performance = np.mean(average_performances)
    results_df.loc[index, "mse"] = average_performance
    print("k = " + str(k))
    print("b = " + str(b))
    print("Average error from 5x2 CV: " + str(average_performance))
    print("")    

# 4. Perform 5x2 cross-validation using the best parameter settings. Average the results of the ten
# experiments and report the average.
best_mse_ind = results_df["mse"].idxmin()
best_k = results_df.loc[best_mse_ind, "k"]
best_bandwidth = results_df.loc[best_mse_ind, "bandwidth"]
print("Best k = " + str(best_k))
print("Best bandwidth = " + str(best_bandwidth))
k = best_k
b = best_bandwidth
average_performances = []
for i in range(5):
    training_data_a, training_data_b = random_split(non_validation_data, "testing")
        
    regressor_a = knn_regressor(k, b, training_data_a, True)
    predictions_a = regressor_a.predict(validation_data, target_col)
    performance_a = evaluate("mse", predictions_a, validation_data, target_col)
    
    regressor_b = knn_regressor(k, b, training_data_b, True)
    predictions_b = regressor_b.predict(validation_data, target_col)
    performance_b = evaluate("mse", predictions_b, validation_data, target_col)

    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("k = " + str(k))
print("bandwidth = " + str(b))
print("Average error from 5x2 CV with best k: " + str(average_performance))
print("")
results_df.to_csv(output_file, index = False)

# 5. Perform 5x2 cross-validation using the null model. Average the results of the ten experiments
# and report the average.
average_performances = []
for i in range(5):
    training_data_a, training_data_b = random_split(non_validation_data, "testing")
    
    regressor_a = null_model(training_data_a[target_col], "regression")
    predictions_a = regressor_a.predict(len(validation_data))
    performance_a = evaluate("mse", predictions_a, validation_data, target_col)
    
    regressor_b = null_model(training_data_b[target_col], "regression")
    predictions_b = regressor_b.predict(len(validation_data))
    performance_b = evaluate("mse", predictions_b, validation_data, target_col)
    
    average_performances.append(performance_a)
    average_performances.append(performance_b)
average_performance = np.mean(average_performances)
print("Average error from 5x2 CV with null model: " + str(average_performance))
print("")
