import numpy as np

class loss_base:
    """
    A base class for loss functions.
    """
    def check_shapes(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def loss_value(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def grad(self):
        raise NotImplementedError("Functionality not implemented for class.")
    
class loss_mse:
    def check_shapes(self, y_pred, y):
        """Checks shapes of predictions and ground truth are consistant.
        
        Expects len(y_pred) == len(y). Data should be: Rows = examples and Columns = outputs/features. Here, examples corresponds to each output node in the last layer - therefore each output represents the gradient wrs that parameter/ mode. 
        
        Example:
        y = np.array([[1],[2]])
        y_pred = np.array([[1], [30]])
        
        loss_mse.delta_MSE() output >> [14.0]

        or
        
        y = np.array([[1, 1], [2, 1]])
        y_pred = np.array([[1, 1], [30, 1]])       
        
        loss_mse.grad() output >> [14.0, 1.0] 
        """
        y_pred = np.array(y_pred)
        y = np.array(y)
        if y_pred.shape != y.shape:
            raise ValueError(f"Shape mismatch: y_pred shape {y_pred.shape} != y shape {y.shape}")

    def loss_value(self, y_pred, y):
        """
        Computes the Mean Squared Error between predictions and ground truth.
        """
        self.check_shapes(y_pred, y)
        return np.mean((y_pred - y) ** 2)

    def loss_grad(self, y_pred, y):
        """
        Computes the gradient of the MSE loss with respect to predictions.
        """
        self.check_shapes(y_pred, y)
        return 2/y_pred.shape[0] * (y_pred - y)