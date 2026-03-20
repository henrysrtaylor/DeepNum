"""Activation functions.

Contains af_relu which applies ReLU or Leaky ReLU element-wise.
Stores pre-activation values for gradient computation.
"""

import numpy as np

class _af_base():
    """
    A base class for activation functions in a neural network.
    """
    def __init__(self):
        self.training = True  # default: training mode
        self.information = {"type": self.__class__.__name__, "parameters": hasattr(self, 'weights') or hasattr(self, 'bias')} 
    def train(self):
        self.training = True
    def eval(self):
        self.training = False
    def forward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def backward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")
    
class af_relu(_af_base):
    """
    ReLu activation function. 
    If leaky_mul set above 0 then leaky ReLu, else ReLu.
    """
    def __init__(self, leaky_mul=0):
        super().__init__()
        self.leaky_mul = np.maximum(0, leaky_mul)
        self.information = {"type": self.__class__.__name__, "parameters": hasattr(self, 'weights') or hasattr(self, 'bias')}
        
    def forward_pass(self, z):
        """
        f(x)=max(0,z)
        """
        def record_layer_inputs(z):
            """
            x is input to the layer during forward pass and used in backprop for the calculation of gradients for a given layer.
            """
            self.inputs = z # record layer mask during forward pass for backpropagation
        record_layer_inputs(z)
        return np.where(z>0, z, z*self.leaky_mul)
        
    def backward_pass(self, error_signal):
        """
        Compute gradients for parameters using local_dev and error_signal.
        local_grad multiplied by error_signal from previous (in this case sequentialy ahead) layer.
        """
        if not hasattr(self, "inputs"):
            raise ValueError("inputs not saved. Ensure forward_pass is called before backward_pass.")
        
        mask = np.where(self.inputs>0, 1, self.leaky_mul).astype(error_signal.dtype) # (B, in_dim)
        dY = error_signal # (B, out_dim)
        new_error_signal = mask * dY # element-wise (B, out_dim)
        return new_error_signal