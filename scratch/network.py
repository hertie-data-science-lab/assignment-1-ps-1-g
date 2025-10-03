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
        
        self.train_accuracies = []
        self.val_accuracies = []


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
        X = np.asarray(x_train, dtype=float)
        if X.ndim == 1:  # single sample
            X = X.reshape(1, -1)
            
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']

        # Layer 1
        Z1 = X @ W1.T
        A1 = self.activation_func(Z1)

        # Layer 2
        Z2 = A1 @ W2.T
        A2 = self.activation_func(Z2)

        # Layer 3
        Z3 = A2 @ W3.T
        Z3 -= np.max(Z3, axis=1, keepdims=True)  # stability
        A3 = self.output_func(Z3.T).T  # softmax outputs

        self.cache = {"X": X, "A1": A1, "A2": A2, "Z1": Z1, "Z2": Z2, "Z3": Z3, "A3": A3}

        return A3


    def _backward_pass(self, y_train, output):
        Y = np.asarray(y_train, dtype=float)
        if Y.ndim == 1: Y = Y.reshape(1, -1)

        X, A1, A2 = self.cache["X"], self.cache["A1"], self.cache["A2"]
        W2, W3 = self.params['W2'], self.params['W3']
        N = X.shape[0]

        S = output # softmax outputs
        # Output gradient (MSE + softmax)
        dL_ds = (S - Y) / N
        SV = S * dL_ds
        s_dot_v = np.sum(SV, axis=1, keepdims=True)
        G3 = SV - S * s_dot_v

        # Backprop hidden layers
        G2 = (G3 @ W3) * (A2 * (1 - A2))
        G1 = (G2 @ W2) * (A1 * (1 - A1))

        return {
            "dW3": G3.T @ A2,
            "dW2": G2.T @ A1,
            "dW1": G1.T @ X
        }

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
        preds = self.predict(x_val)
        if preds.ndim == 0:  # single sample
            return float(preds == np.argmax(y_val))
        return np.mean(preds == np.argmax(y_val, axis=1))

    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.
        The method should return the index of the most likeliest output class.
        '''
        probs = self._forward_pass(x)
        return np.argmax(probs, axis=1) if probs.ndim == 2 else int(np.argmax(probs))

    

    def fit(self, x_train, y_train, x_val, y_val, cosine_annealing_lr=False, batch_size=32):
        start_time = time.time()
        num_samples = x_train.shape[0]
        total_steps = (num_samples // batch_size) * self.epochs

        for epoch in range(self.epochs):
            # shuffle training data each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_train, y_train = x_train[indices], y_train[indices]

            for step in range(0, num_samples, batch_size):
                xb = x_train[step:step+batch_size]
                yb = y_train[step:step+batch_size]

                # learning rate scheduler
                if cosine_annealing_lr:
                    global_step = epoch * (num_samples // batch_size) + (step // batch_size)
                    lr = cosine_annealing(self.learning_rate,
                                        global_step,
                                        total_steps,
                                        min_lr=0.0)
                else:
                    lr = self.learning_rate

                # forward + backward + update
                output = self._forward_pass(xb)
                grads = self._backward_pass(yb, output)
                self._update_weights(grads, lr)

            train_acc = self.compute_accuracy(x_train, y_train)
            val_acc = self.compute_accuracy(x_val, y_val)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            # print epoch progress
            self._print_learning_progress(start_time, epoch, x_train, y_train, x_val, y_val)

            
