import time
import numpy as np
import scratch.utils as utils
from scratch.lr_scheduler import cosine_annealing


class Network():
    def __init__(self, sizes, epochs=50, learning_rate=0.01, random_state=1):
        self.sizes = sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.activation_func = utils.sigmoid
        self.activation_func_deriv = utils.sigmoid_deriv
        self.output_func = utils.softmax
        self.output_func_deriv = utils.softmax_deriv
        self.cost_func = utils.mse
        self.cost_func_deriv = utils.mse_deriv

        self.params = self._initialize_weights()


    def _initialize_weights(self):
        # number of neurons in each layer
        input_layer = self.sizes[0]
        hidden_layer_1 = self.sizes[1]
        hidden_layer_2 = self.sizes[2]
        output_layer = self.sizes[3]

        # random initialization of weights
        np.random.seed(self.random_state)
        params = {
            'W1': np.random.rand(hidden_layer_1, input_layer) - 0.5,
            'W2': np.random.rand(hidden_layer_2, hidden_layer_1) - 0.5,
            'W3': np.random.rand(output_layer, hidden_layer_2) - 0.5,
        }

        return params


    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.

        The method should return the output of the network.
        '''
        params=self.params

        # Layer 1
        z1=np.dot(params['W1'], x_train.T)
        a1=self.activation_func(z1)

        # Layer 2
        z2=np.dot(params['W2'], a1)
        a2=self.activation_func(z2)

        #Layer 3
        z3=np.dot(params['W3'], a2)
        a3=self.output_func(z3)

        #Final output
        output={"x":x_train,
                "z1": z1, "a1":a1,
                "z2": z2, "a2":a2,
                "z3":z3, "a3":a3}
        return a3, output


    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.

        The method should return a dictionary of the weight gradients which are used to update the weights in self._update_weights().

        '''
        m=y_train.shape[0]
        params=self.params

        x_train=output["x"]
        a1,a2,a3=output["a1"], output["a2"], output["a3"]
        z1,z2,z3= output["z1"], output["z2"], output["z3"]

        #Output layer
        dz3=(a3-y_train.T)*self.output_func_deriv(z3)
        dW3=(1/m)*np.dot(dz3,a2.T)

        #Hidden Layer 2
        dz2=np.dot(params['W3'].T, dz3)* self.activation_func_deriv(z2)
        dW2 = (1/m) * np.dot(dz2, a1.T)

        # Hidden layer 1
        dz1 = np.dot(params['W2'].T, dz2) * self.activation_func_deriv(z1)
        dW1 = (1/m) * np.dot(dz1, x_train)
        
        weights_gradient = {"dW1": dW1, "dW2": dW2, "dW3": dW3}
        return weights_gradient


    def _update_weights(self, weights_gradient, learning_rate):
        '''
        TODO: Update the network weights according to stochastic gradient descent.
        '''
        for l in [1,2,3]:
            self.params[f'W{l}'] -= learning_rate * weights_gradient[f'dW{l}']


    def _print_learning_progress(self, start_time, iteration, x_train, y_train, x_val, y_val):
        train_accuracy = self.compute_accuracy(x_train, y_train)
        val_accuracy = self.compute_accuracy(x_val, y_val)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def compute_accuracy(self, x_val, y_val):
        predictions = []
        for x, y in zip(x_val, y_val):
            pred = self.predict(x)
            predictions.append(pred == np.argmax(y))

        return np.mean(predictions)


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        a3, _=self._forward_pass(x)
        return np.argmax(a3, axis=0)



    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False):

        start_time = time.time()

        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                x=x.reshape(1,-1)
                y=y.reshape(1,-1)
                
                if cosine_annealing_lr:
                    learning_rate = cosine_annealing(self.learning_rate, 
                                                     iteration, 
                                                     self.epochs, 
                                                     min_lr=0)
                else: 
                    learning_rate = self.learning_rate
                output = self._forward_pass(x)
                weights_gradient = self._backward_pass(y, output)
                
                self._update_weights(weights_gradient, learning_rate=learning_rate)

            self._print_learning_progress(start_time, iteration, x_train, y_train, x_val, y_val)

if __name__ == "__main__":
    from scratch.network import Network
    import numpy as np

    net = Network([784, 128, 64, 10])
    X = np.random.randn(5, 784)  # dummy batch of 5 images
    out, _ = net._forward_pass(X)
    print("Forward output shape:", out.shape)