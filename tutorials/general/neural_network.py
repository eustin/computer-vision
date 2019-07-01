
import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        # layers = list of integers defining the architecture of network.
        #          e.g. [2, 2, 1] means input layer and hidden layer with two
        #               nodes and an output of one node.
        self.W = []
        self.layers = layers
        self.alpha = alpha

        for i in range(len(layers) - 2):
            # create M + 1 x N + 1 weight matrix where M is the number of nodes in current layer
            # and N is number of nodes in the next layer and the 1s represent the bias terms
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            # denominator is a scaling term applied to the weights
            self.W.append(w / np.sqrt(layers[i]))

        # handle the last two layers. The input into the last layer requires a bias term
        # but the output does not.
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        # this is the derivative of the sigmoid function
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display_update=100):
        X = np.c_[X, np.ones(X.shape[0])]
        for epoch in range(epochs):
            for x, target in zip(X, y):
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print("epoch={}, loss{:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        # this is where post-activation values are calculated for each layer.
        # we store these values in a list, A
        A = [np.atleast_2d(x)]

        # forward pass
        for layer in range(len(self.W)):
            preactivation = A[layer].dot(self.W[layer])
            postactivation = self.sigmoid(preactivation)
            A.append(postactivation)

        # calculate errors using postactivation values in output layer and ground
        # truth labels
        error = A[-1] - y

        # backward pass
        D = [error * self.sigmoid_derivative(A[-1])]
        # ignore the last two layers as we have already calculated their deltas in
        # the previous statement
        for layer in np.arange(len(A) - 2, 0, -1):
            # delta of current layer is equal to delta of previous layer dotted with
            # weight matrix of current layer
            delta = D[-1].dot(self.W[layer].T)
            # multiply the delta of the current layer by the delta of the activations
            # in the current layer
            delta = delta * self.sigmoid_derivative(A[layer])
            D.append(delta)

        # gradient descent (i.e. weight update step)
        # reverse the deltas as we worked through them backwards
        D = D[::-1]
        for layer in range(len(self.W)):
            self.W[layer] = self.W[layer] - self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, add_bias=True):
        p = np.atleast_2d(X)

        if add_bias:
            p = np.c_[p, np.ones(p.shape[0])]

        # take dot product of input into current layer with weights matrix in
        # current layer
        for layer in range(len(self.W)):
            p = np.dot(p, self.W[layer])
            p = self.sigmoid(p)

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, add_bias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        return loss

    def __repr__(self):
        # print network architecture
        return "NN architecture: {}".format("-".join(str(l) for l in self.layers))
