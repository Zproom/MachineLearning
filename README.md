# MachineLearning

## Overview

This project implements three machine learning algorithms from scratch and evaluates their performance across multiple datasets from the UC Irvine Machine Learning Repository. The algorithms implemented are k-nearest neighbors (KNN), decision trees, and neural networks. The project uses cross-validation to find optimal hyperparameters for each algorithm on each dataset. This work was completed as part of my master's degree in computer science at Johns Hopkins University.

## How to run the code

Each algorithm is organized in its own directory: `decisiontree/`, `knn/`, and `neuralnet/`. To run the experiments and find optimal hyperparameters for each algorithm:

1. Navigate to the desired algorithm directory: `cd knn/code/`, `cd decisiontree/code/`, or `cd neuralnet/code/`
2. Run the appropriate experiment file:
   - KNN: `python experiments_regular_knn.py` or `python experiments_edited_knn.py`
   - Decision Tree: `python experiments.py`
   - Neural Network: `python experiments.py`

The experiment files use cross-validation to evaluate different hyperparameter configurations and save the results to .csv files in the `data/` directory. The R script `decisiontree/code/analysis.R` can be used to generate convergence plots from the decision tree pruning results.

## Technologies Used

Most of the codebase is written in Python (Python 3.12). The Python code only uses `pandas` and `numpy`. Some analytical code is written in R (R 4.4.2). The packages used in the R code include `tidyverse` and `this.path`.

## Data Source

All datasets are sourced from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). The six datasets used in this project are:

1. Abalone: Regression task predicting the age of abalone from physical characteristics
2. Breast Cancer Wisconsin: Binary classification task for diagnosing breast cancer
3. Car Evaluation: Multi-class classification task for evaluating car acceptability
4. Forest Fires: Regression task predicting forest fire area burned
5. House Votes 84: Binary classification task predicting Congressional voting patterns
6. Machine: Regression task predicting CPU runtime performance

## Methods

**K-Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies or regresses data points based on the majority class or average value of the k nearest neighbors. Both regular and edited KNN variants are implemented, with the latter removing redundant training examples.

**Decision Trees**: A tree-based model built using the ID3 algorithm that recursively partitions the feature space. The implementation includes support for both classification and regression, with reduced error pruning to prevent overfitting.

**Neural Networks**: Feedforward neural networks trained with backpropagation for both classification and regression tasks. The implementation includes logistic regression and linear regression as foundational algorithms. The project compares two approaches: (1) standard feedforward networks trained directly on classification/regression tasks, and (2) autoencoder-based architectures that use unsupervised pre-training for feature extraction.

## Future Work

Potential enhancements to this project include:

- Implementing additional algorithms such as reinforcement learning (work in progress), support vector machines (SVM), or ensemble methods
- Optimizing code performance for larger-scale datasets
- Creating visualization tools to compare algorithm performance across datasets