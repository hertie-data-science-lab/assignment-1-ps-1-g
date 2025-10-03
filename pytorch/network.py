import time

import torch
import torch.nn as nn
import torch.optim as optim


class TorchNetwork(nn.Module):
    def __init__(self, sizes, epochs=10, learning_rate=0.01, random_state=1):
        super().__init__()

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(self.random_state)

        self.linear1 = nn.Linear(sizes[0], sizes[1])
        self.linear2 = nn.Linear(sizes[1], sizes[2])
        self.linear3 = nn.Linear(sizes[2], sizes[3])

        self.activation_func = torch.sigmoid
        self.output_func = torch.softmax
        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)

        self.train_accuracies = []
        self.val_accuracies = []



    def _forward_pass(self, x_train):
        '''
        TODO: Implement the forward propagation algorithm.
        The method should return the output of the network.

        Froward propagation: input, hidden1 sigmoid, hidden2 sigmoid, output softmax
        '''
        h1 = self.activation_func(self.linear1(x_train))
        h2 = self.activation_func(self.linear2(h1))
        logits = self.linear3(h2)

        return logits



    def _backward_pass(self, y_train, output):
        '''
        TODO: Implement the backpropagation algorithm responsible for updating the weights of the neural network.
        '''
        y_train = y_train.float() #make sure labels are float for MSE
        loss = self.loss_func(output, y_train)
        loss.backward()
        return loss.item()


    def _update_weights(self):
        '''
        TODO: Update the network weights according to stochastic gradient descent.
        '''
        self.optimizer.step()


    def _flatten(self, x):
        return x.view(x.size(0), -1)       


    def _print_learning_progress(self, start_time, iteration, train_loader, val_loader):
        train_accuracy = self.compute_accuracy(train_loader)
        val_accuracy = self.compute_accuracy(val_loader)
        print(
            f'Epoch: {iteration + 1}, ' \
            f'Training Time: {time.time() - start_time:.2f}s, ' \
            f'Learning Rate: {self.optimizer.param_groups[0]["lr"]}, ' \
            f'Training Accuracy: {train_accuracy * 100:.2f}%, ' \
            f'Validation Accuracy: {val_accuracy * 100:.2f}%'
            )


    def predict(self, x):
        '''
        TODO: Implement the prediction making of the network.

        The method should return the index of the most likeliest output class.
        '''
        self.eval()
        x = self._flatten(x)
        with torch.no_grad():
            logits = self._forward_pass(x)
            probs = torch.softmax(logits, dim=1)   # use softmax for multiclass
        return torch.argmax(probs, dim=1)


    def fit(self, train_loader, val_loader):
        start_time = time.time()

        for epoch in range(self.epochs):
            for x, y in train_loader:
                x = x.view(x.size(0), -1)

                logits = self._forward_pass(x)
                y = nn.functional.one_hot(y, num_classes=10).float()  # convert to one-hot
                loss = self.loss_func(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_acc = self.compute_accuracy(train_loader)
            val_acc = self.compute_accuracy(val_loader)

            # store accuracies
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self._print_learning_progress(start_time, epoch, train_loader, val_loader)


    def compute_accuracy(self, data_loader):
        self.eval()
        correct, total = 0,0
        with torch.no_grad():
            for x, y in data_loader:
                preds = self.predict(x)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return correct / total
