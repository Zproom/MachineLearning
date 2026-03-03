# null_models.py
# This file defines a class to implement null models for both classification and regression.

import pandas as pd
import numpy as np
import random

class null_model:
    def __init__(self, labels, type):
        """This method initializes a null_model instance.

        Args:
            labels (pandas.Series): The labelled outputs from the training data.
            type (str): The type of null model to be implemented ("classification" or "regression").
        """
        if type != "classification" and type != "regression":
            raise Exception("type must be either 'classification' or 'regression'.")
     
        self.labels = labels
        self.type = type
    
    def predict(self, n_predictions):
        """This method returns the null_model prediction.

        Args:
            n_predictions (int): The number of predictions needed. This is the number of 
            observations in the test set.

        Returns:
            (pd.Series): A prediction for every new example. 
        """
        # Find the mode.
        if self.type == "classification":
            prediction = self.labels.mode()
            # Choose randomly if there are ties.
            if len(prediction) > 1:
                prediction = random.choice(prediction.index.tolist())
            return pd.Series(np.repeat(prediction, 
                                       n_predictions)).reset_index(drop = True)
        # Find the average.
        else:
            prediction = self.labels.mean()
            return pd.Series(np.repeat(prediction, n_predictions)).reset_index(drop = True)