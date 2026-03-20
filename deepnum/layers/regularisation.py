"""Regularisation layers like dropout.

Contains reg_dropout which randomly zeros activations during training to reduce overfitting.
"""

import numpy as np

class _reg_base():
    """
    A base class for regularisation functions in a neural network.
    """
    def __init__(self):
        self.training = True
        self.information = {"type": self.__class__.__name__, "parameters": hasattr(self, 'weights') or hasattr(self, 'bias')}   
    def train(self):
        self.training = True
    def eval(self):
        self.training = False
    def forward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def backward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")

class reg_dropout(_reg_base):
    """
    Dropout layer acting on inputs to randomly drop activations from previous layer for regularisation.
    """
    def __init__(self, p_drop: float = 0.2):
        super().__init__()
        if not (0.0 <= p_drop < 1.0):
            raise ValueError("p_drop must be in [0, 1).")

        self.p_drop = p_drop
        self.p_keep = 1 - self.p_drop
        
    def forward_pass(self, x):
        if self.training == False or self.p_keep == 1:
            self.mask = None # set mask to None
            return x
        
        mask = np.random.binomial(n=1, p=self.p_keep, size=x.shape) # Bernoulli mask (keep)
        self.mask = mask # Save mask for backprop.
        self.mask = self.mask.astype(x.dtype)
        
        # Inverted dropout: during training scale by 1/p_keep so E[output] matches inference (no dropout) in signal
        # in other words normalising the surviving activations so that they carry the same total signal as if nothing had been dropped ie still signal == 1.
        return (x * self.mask) / self.p_keep
    
    def backward_pass(self, error_signal):
        if self.mask is None:
            return error_signal 
            
        # Backward mirrors forward: gradients are masked and scaled by the same factor
        return (self.mask / self.p_keep) * error_signal