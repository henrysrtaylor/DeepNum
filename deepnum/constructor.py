"""Model classes for building and running neural networks.

Contains sequential_model which holds layers and handles forward/backward passes through the network.
"""

from typing import Callable

class _model_base():
    """
    Sequential model to hold layers and calculate forward and back passes.
    """
    def __init__(self, layers: list):
        self.layers = layers
        self.training = True  # default: training mode
    def train(self):
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
    def eval(self):
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
    def forward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def backward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def zero_grad(self):
        """
        Zeros local derivatives to free memory and ensure no stale gradients are used.
        """
        for layer in self.layers:
            if hasattr(layer, 'inputs'):
                delattr(layer, 'inputs')   

class sequential_model(_model_base):
    """
    Sequential model to hold layers and calculate forward and back passes.
    """
    def __init__(self, layers: list):
        super().__init__(layers)
        
    def forward_pass(self, x):
        """
        Forward pass of sequential model.
        """
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x
    
    def backward_pass(self, error_signal, parameter_update_function: Callable):
        """
        Backward pass of sequential model - backpropagation and weight update rule according to parameter_update_function from optimiser.
        """
        for layer in reversed(self.layers):
            # update if layer has weights (parameters)
            error_signal = layer.backward_pass(error_signal, parameter_update_function) if hasattr(layer, "weights") else layer.backward_pass(error_signal)
    