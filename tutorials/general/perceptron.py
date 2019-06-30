
import numpy as np


class Perceptron:

    def __init__(self, N, alpha=0.1):
        # one extra weight for bias term
        # divide by sqrt of N to scale our weight matrix
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    @staticmethod
    def step(x):
        # define step function
        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        X = np.c_[X, np.ones(X.shape[0])]

        for epoch in range(epochs):
            for x, target in zip(X, y):
                dot_prod = np.dot(x, self.W)
                prediction = self.step(dot_prod)

                # if prediction not equal to target label, make update
                if prediction != target:
                    error = prediction - target
                    self.W = self.W - self.alpha * error * x

    def predict(self, X, add_bias=True):
        X = np.atleast_2d(X)
        if add_bias:
            X = np.c_[X, np.ones(X.shape[0])]
        return self.step(np.dot(X, self.W))



