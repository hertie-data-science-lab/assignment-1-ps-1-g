import time
import numpy as np
from scratch.network import Network

class ResNetwork(Network):


    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        '''
        TODO: Initialize the class inheriting from scratch.network.Network.
        The method should check whether the residual network is properly initialized.
        '''
        super().__init__(sizes, epochs, learning_rate, random_state)

        # Check if residual connection is properly initialised
        if sizes[1] != sizes[2]:
            raise ValueError("Residual connection requires hidden_layer_1 and hidden_layer_2 to have the same size")
        


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.
        '''
        params = self.params
        X = np.asarray(x_train, dtype=float)

        # Ensure 2D shape (batch, features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Layer 1
        z1 = X @ params['W1'].T          
        a1 = self.activation_func(z1)

        # Layer 2 with residual skip
        z2 = a1 @ params['W2'].T         
        a2 = self.activation_func(z2) + a1  # add skip

        # Output
        z3 = a2 @ params['W3'].T
        z3 -= np.max(z3, axis=1, keepdims=True)  # stability
        a3 = self.output_func(z3.T).T

        # store cache for backprop
        self.cache = {"X": X, "z1": z1, "a1": a1,
                     "z2": z2, "a2": a2,
                     "z3": z3, "a3": a3}

        return a3


    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
        The method should also account for the residual connection in the hidden layer.

        '''
        params = self.params
        X, a1, a2 = self.cache["X"], self.cache["a1"], self.cache["a2"]
        z1, z2, a3 = self.cache["z1"], self.cache["z2"], self.cache["a3"]

        m = y_train.shape[0]
        Y = np.asarray(y_train, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(1, -1)

        # Output error
        dz3 = (a3 - Y) / m                
        dW3 = dz3.T @ a2                 

        # Hidden layer 2
        da2 = dz3 @ params['W3']          
        dz2 = da2 * self.activation_func_deriv(z2)
        dW2 = dz2.T @ a1                 

        # Hidden layer 1 (two paths: normal + skip)
        da1 = dz2 @ params['W2'] + da2    
        dz1 = da1 * self.activation_func_deriv(z1)
        dW1 = dz1.T @ X                  

        return {"dW1": dW1, "dW2": dW2, "dW3": dW3}

