"""Loss functions for training.

Provides loss_mse (mean squared error) for regression,
and loss_cross_entropy for multi-class classification.
"""

import numpy as np

def _check_shapes(y_pred, y):
    """Checks shapes of predictions and ground truth are consistant.
    
    Expects len(y_pred) == len(y). Data should be: Rows = examples and Columns = outputs/features. 
    
    Here, examples corresponds to each output node in the last layer - therefore each output represents the gradient wrs that parameter/ mode. 
    """
    y_pred = np.array(y_pred)
    y = np.array(y)
    if y_pred.shape != y.shape:
        raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y shape {y.shape}")

class _loss_base:
    """
    A base class for loss functions.
    """
    def loss_value(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def grad(self):
        raise NotImplementedError("Functionality not implemented for class.")

class loss_cross_entropy(_loss_base):
    """
    Cross entropy loss for multi-class classification (takes raw logits).
    *DO NOT USE SOFTMAX LAYER IF USING THIS LOSS FUNCTION*
    
    Expects:
        logits: raw logits from network before softmax (B, C)
        y: one-hot encoded labels (B, C)
    
    L = -sum(y * log(softmax(logits))) averaged over batch
    
    """
    def __init__(self):
        super().__init__()

    def loss_value(self, logits, y):
        """
        Computes cross entropy using log-sum-exp trick for stability.
        
        Trick 1 (log-softmax): log(softmax(z)) = z - log(sum(exp(z)))
            - Cross entropy only needs log(p), so we substitute
            - Trick: log(softmax(z)) = z - log(sum(exp(z))) (proven identity)
            - This trick only works for when we do Cross Entropy after.
            - Avoids: log(0) when softmax would output tiny values

        Trick 2 (stability, as in AF Softmax): z_stable = z - max(z)  
            - Prevents overflow in exp() for large logits
            - Avoids: exp(1000) = inf
        """
        _check_shapes(logits, y)
        z_stable = logits - np.max(logits, axis=1, keepdims=True) # stability: subtract max per row so that max becomes 0. Each row is a observation for a batch. exp([1000, 1001, 1002]) = [inf, inf, inf] |  exp([-2, -1, 0]) = [0.135, 0.368, 1.0] 
        log_softmax = z_stable - np.log(np.sum(np.exp(z_stable), axis=1, keepdims=True)) # log-softmax = z - log(sum(exp(z))), more stable than log(exp(z)/sum(exp(z)))
        return -np.mean(np.sum(y * log_softmax, axis=1)) # Cross entropy: -sum(y * log(p)), averaged over batch

    def loss_grad(self, logits, y):
        """
        Gradient of cross entropy wrt logits is much simpler:
            dL/dz = softmax(z) - y
        
        Essentially, this is because both functions can be simplified by taking parts of both
        
        Derivation (chain rule with softmax and cross entropy):
            Cross entropy:
                L = -sum(y * log(p))  where p = softmax(z)
            
                dL/dp = -y / p --- gradient of CE wrt probs
                dp/dz = p * (I - p^T) --- Jacobian for Softmax - more complex
                
            Combined:
                dL/dz = dL/dp * dp/dz = p - y  --- where p is probability and y is ground truth, much simplier
        
        Intuition:
            - If true class is 0: y = [1, 0, 0], p = [0.7, 0.2, 0.1]
            - Gradient = [0.7-1, 0.2-0, 0.1-0] = [-0.3, 0.2, 0.1]
            - Push true class UP (negative gradient), wrong classes DOWN (positive gradient)
            
        This simplification is why we combine softmax into the loss rather than using separate layers.
        """
        _check_shapes(logits, y)
        softmax = self._softmax(logits)
        return (softmax - y) / logits.shape[0]  # divide by batch size for mean
    
    def _softmax(self, logits):
        """       
        Stability: subtract max per row so max becomes 0.
        exp([1000, 1001, 1002]) = [inf, inf, inf] | exp([-2, -1, 0]) = [0.135, 0.368, 1.0]
        
        See activation functions Softmax for more details
        """
        z_stable = logits - np.max(logits, axis=1, keepdims=True)  # shift so max = 0, prevents overflow
        exp_z = np.exp(z_stable)  # now safe: max value is exp(0) = 1
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # (B, C) probabilities summing to 1

class loss_mse(_loss_base):
    """
    Loss mean squared error for regression.
    Example:
        y = np.array([[1],[2]])
        y_pred = np.array([[1], [30]])
        
        loss_mse.delta_MSE() output >> [14.0]

        or
        
        y = np.array([[1, 1], [2, 1]])
        y_pred = np.array([[1, 1], [30, 1]])       
        
        loss_mse.grad() output >> [14.0, 1.0] 
    """
    def __init__(self):
        super().__init__()
     
    def loss_value(self, y_pred, y):
        """
        Computes the Mean Squared Error between predictions and ground truth.
        """
        _check_shapes(y_pred, y)
        return np.mean((y_pred - y) ** 2)

    def loss_grad(self, y_pred, y):
        """
        Computes the gradient of the MSE loss with respect to predictions.
        """
        _check_shapes(y_pred, y)
        return 2/y_pred.shape[0] * (y_pred - y)