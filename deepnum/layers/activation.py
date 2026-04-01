"""Activation functions.

Contains af_relu (ReLU/Leaky ReLU), af_softmax (multi-class), and af_sigmoid (binary).
Stores outputs/inputs for gradient computation during backpropagation.
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
        return mask * dY # element-wise (B, out_dim)
    
class af_softmax(_af_base):
    """
    Softmax activation function. 
    """
    def __init__(self):
        super().__init__()
        self.information = {"type": self.__class__.__name__, "parameters": hasattr(self, 'weights') or hasattr(self, 'bias')}
        
    def forward_pass(self, z):
        """
        p_i = exp(z_i) / sum(exp(z))
        """
        z_stable = z - np.max(z, axis=1, keepdims=True) # stability: subtract max per row so that max becomes 0. Each row is a observation for a batch. exp([1000, 1001, 1002]) = [inf, inf, inf] |  exp([-2, -1, 0]) = [0.135, 0.368, 1.0] 
        exp_z = np.exp(z_stable) # exp for each element in array
        self.outputs = exp_z / np.sum(exp_z, axis=1, keepdims=True) # input (B, C) > output (B, C)
        return self.outputs
        
    def backward_pass(self, error_signal):
        """
        Softmax gradient: dL/dz_i = p_i * (dL/dp_i - sum_j(p_j * dL/dp_j))
        """
        if not hasattr(self, "outputs"):
            raise ValueError("outputs not saved. Ensure forward_pass is called before backward_pass.")
        s = self.outputs  # (B, out_dim) softmax probabilities from forward pass
        dY = error_signal  # (B, out_dim) gradient coming from the loss
        
        # (B, out_dim) weight each gradient by its probability and sum. Weighted average gradient which is a correction term
        sum_term  = np.sum(dY * s, axis=1, keepdims=True) 
        
        # how different is each class's gradient from the average? aka center the gradients | then scale by probability - classes with higher probability get more gradient and rare classes get less
        new_error_signal = s * (dY - sum_term)
        return new_error_signal
    
class af_sigmoid(_af_base):
    """
    Sigmoid activation function. 
    """
    def __init__(self):
        super().__init__()
        self.information = {"type": self.__class__.__name__, "parameters": hasattr(self, 'weights') or hasattr(self, 'bias')}
        
    def forward_pass(self, z):
        """
        s(z) = 1 / (1 + exp(-z))
        """
        z_clipped = np.clip(z, -500, 500) # stability: clip to prevent overflow: exp(-z) overflows for large negative z. As sigmoid approaches these values, they approach 0, 1 respectively so clipping makes no difference to output.
        self.outputs = 1 / (1 + np.exp(-z_clipped)) # single probability output
        return self.outputs
    
    def backward_pass(self, error_signal):
        """
        gradient: dL/dz = dL/ds * ds/dz = dL/ds * s * (1 - s)
        
        derivative: ds/dz = s(z) * (1 - s(z))
        
        Intuition: gradient peaks at s=0.5 (uncertain) and vanishes at s=0 or s=1 (confident).
        """
        if not hasattr(self, "outputs"):
            raise ValueError("outputs not saved. Ensure forward_pass is called before backward_pass.")
        s = self.outputs  # (B, out_dim) sigmoid output
        dY = error_signal  # (B, out_dim) gradient from loss: dL/ds
        
        # Chain rule: dL/dz = dL/ds * ds/dz = dY * s * (1 - s)
        return dY * s * (1 - s) # eliminates if s close to 0 or 1