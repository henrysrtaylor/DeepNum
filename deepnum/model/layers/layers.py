import numpy as np
from typing import Callable

class layer_base():
    """
    A base class for layers in a neural network.
    """
    def __init__(self, num_nodes_input, num_nodes_output):
        self.num_input_node = num_nodes_input
        self.num_output_node = num_nodes_output
        self.information = {"type": self.__class__.__name__, "parameters": hasattr(self, 'weights') or hasattr(self, 'bias')}
    def forward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")
    def backward_pass(self):
        raise NotImplementedError("Functionality not implemented for class.")
    
class layer_linear(layer_base):
    """
    A linear layer in a neural network.
    """
    def __init__(self, num_nodes_input, num_nodes_output):
        super().__init__(num_nodes_input, num_nodes_output)
        self.weights = np.random.uniform(-np.sqrt(6 / (num_nodes_input+num_nodes_output)), np.sqrt(6 / (num_nodes_input+num_nodes_output)), (num_nodes_input, num_nodes_output)) # Xavier uniform initialization - more stable than random 
        # self.weights = np.random.rand(num_nodes_input, num_nodes_output) # random init weights
        self.bias = np.random.rand(1, num_nodes_output)
        self.information = {"type": self.__class__.__name__, "parameters": hasattr(self, 'weights') or hasattr(self, 'bias')}
        
    def forward_pass(self, x):
        """
        forward pass: y=Wx+b
        arrays: row x col
        shape: nm x ij = nj
        """
        def record_layer_inputs(x):
            """
            x is input to the layer during forward pass and used in backprop for the calculation of gradients for a given layer.
            """
            self.inputs = x

        x = np.array(x)
        
        if x.shape[1] != self.weights.shape[0]:
            raise ValueError(
                f"Shape mismatch between input and weights: input has {x.shape[1]} features, but weights expect {self.weights.shape[0]}."
            )

        record_layer_inputs(x) # record layer inputs during forward pass for backpropagation
        
        y = np.matmul(x, self.weights) + self.bias # bias will be 0 if not enabled
        return y
    
    def backward_pass(self, error_signal, parameter_update_rule: Callable):
        """
        Backpropagation Algorithm.
        """
        if not hasattr(self, "inputs"):
            raise ValueError("inputs not saved. Ensure forward_pass is called before backward_pass.")
        
        # matrix form
        X = self.inputs # (B, in_dim)
        W = self.weights # (in_dim, out_dim)
        B = self.bias # (out_dim,)
        dY = error_signal # (B, out_dim)
        
        # gradient wrs weights
        grad_wrt_w = np.matmul(X.T, dY)
        
        # gradient wrs bias
        grad_wrt_b = np.sum(dY, axis=0) # Axis = 0 will sum over batch meaning that we get shape of (out_dim,).
        
        # update weights and bias
        new_weights = parameter_update_rule(old_parameters=W, grad=grad_wrt_w)
        new_bias = parameter_update_rule(old_parameters=B, grad=grad_wrt_b)
        self.weights=new_weights
        self.bias=new_bias
        
        # calculate layer error signal for layer - this is just error signal multiplied by weights transposed
        new_error_signal = np.matmul(dY, W.T)
        
        return new_error_signal