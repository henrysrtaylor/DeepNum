"""Common evaluation metrics.

Classification: metric_accuracy, metric_precision, metric_recall, metric_f1
Regression: metric_mse, metric_rmse, metric_mae
"""

import numpy as np

# HELPER
def _validate_inputs(y_pred, y_true):
    """
    Validates metric inputs are numpy arrays and the correct shape.
    """
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be a numpy array")
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be a numpy array")
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Input shapes must match: y_pred {y_pred.shape} != y_true {y_true.shape}")

# CLASSIFICATION
def metric_accuracy(y_pred_prob, y_true):
    """
    Accuracy - proportion of correct predictions.
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN) = correct / total
    
    Input:
        y_pred: logits or probabilities (B, C)
        y_true: one-hot encoded labels (B, C)
    
    Output:
        accuracy - float between 0 and 1
    """
    _validate_inputs(y_pred_prob, y_true)
    pred_classes = np.argmax(y_pred_prob, axis=1) # get predicted class position on each row (observation).
    true_classes = np.argmax(y_true, axis=1) # get ground truth class for each observation.
    return np.mean(pred_classes == true_classes)

def metric_recall(y_pred, y_true):
    """
    
    """
    _validate_inputs(y_pred, y_true)

def metric_precision(y_pred, y_true):
    """
    
    """
    _validate_inputs(y_pred, y_true)

def metric_f1(y_pred, y_true):
    """
    
    """
    _validate_inputs(y_pred, y_true)

# REGRESSION
def metric_mse(y_pred, y_true):
    """
    
    """
    _validate_inputs(y_pred, y_true)

def metric_rmse(y_pred, y_true):
    """
    
    """
    _validate_inputs(y_pred, y_true)

def metric_mae(y_pred, y_true):
    """
    
    """
    _validate_inputs(y_pred, y_true)