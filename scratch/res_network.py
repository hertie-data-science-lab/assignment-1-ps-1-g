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
        
        # Layer 1
        z1 = np.dot(params['W1'], x_train.T)
        a1 = self.activation_func(z1)
        
        # Layer 2 with residual connection
        z2 = np.dot(params['W2'], a1)
        a2 = self.activation_func(z2) + a1   # residual skip
        
        # Output layer
        z3 = np.dot(params['W3'], a2)
        a3 = self.output_func(z3)

        output = {"x":x_train,
                  "z1": z1, "a1": a1,
                  "z2": z2, "a2": a2,
                  "z3": z3, "a3": a3}
        return a3, output



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().
        The method should also account for the residual connection in the hidden layer.

        '''
        m = y_train.shape[0]
        params = self.params

        x_train = output["x"]
        a1, a2, a3 = output["a1"], output["a2"], output["a3"]
        z1, z2, z3 = output["z1"], output["z2"], output["z3"]

        # Output layer
        dz3 = (a3 - y_train.T) * self.output_func_deriv(z3)
        dW3 = (1/m) * np.dot(dz3, a2.T)

        # Hidden layer 2 (with residual)
        dz2 = np.dot(params['W3'].T, dz3) * self.activation_func_deriv(z2)
        dW2 = (1/m) * np.dot(dz2, a1.T)

        # Hidden layer 1 (residual gradient adds directly)
        dh1_total = np.dot(params['W2'].T, dz2) + dz2
        dz1 = dh1_total * self.activation_func_deriv(z1)
        dW1 = (1/m) * np.dot(dz1, x_train)

        grads = {"dW1": dW1, "dW2": dW2, "dW3": dW3}
        return grads



