# node.py
# This file defines a node class that is used to build all the decision tree models.

class node:
    def __init__(self, feature = None, numeric_split = None, children = None, 
                 pruned_prediction = None, prediction = None):
        """This method initializes a node instance.

        Args:
            feature (str, optional): The name of the feature in the DataFrame. Defaults to None.
            numeric_split (float, optional): If the feature is numeric, this is the split point.
            children (dict, optional): The child nodes of this node. Defaults to None. If the
            feature is numeric, the key is either "below" or "above", which means the child is
            below or above numeric_split. If the feature is categorical, the key is a unique
            feature value.
            pruned_prediction (str, optional): If the node is a non-leaf node, this is the
            predicted value to use if the node gets pruned. It is the most common class 
            (classification) or the average target value (regression) in the current partition.
            prediction (str, optional): If this is a leaf node, this is the predicted value. 
            Defaults to None (non-leaf node).
        """
        self.feature = feature
        self.numeric_split = numeric_split
        self.children = children
        self.pruned_prediction = pruned_prediction
        # Only applies to leaf nodes.
        self.prediction = prediction