class optimiser_sgd():
    """
    Operates on model to update weights based on loss function.
    """
    def __init__(self, loss: object, learning_rate: float):
        self.loss = loss
        self.learning_rate = learning_rate
            
    def calculate_gradient(self, y_pred, y):
        """
        Calculates gradient based on loss.
        """
        return self.loss.loss_grad(y_pred, y)
    
    def backward_pass(self, model, y_pred, y):
        """
        Updates model parameters based on calculated gradients.
        """
        def parameter_update_rule(old_parameters, grad):
            """
            Update rules for parameters using gradient descent.
            """           
            if old_parameters is not None:
                new_parameters = old_parameters - (self.learning_rate * grad)
            return new_parameters
                
        # calculate gradients in model
        error_signal = self.calculate_gradient(y_pred, y) # calculate initial gradient from loss
        model.backward_pass(error_signal, parameter_update_rule) # backprop and update according to parameter_update_rule defined in optimiser.  