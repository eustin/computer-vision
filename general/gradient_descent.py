
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))


def predict(X, W):
    preds = sigmoid_activation(X.dot(W))
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1
    return preds


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

# generate some data
# every row in array X has two columns
X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
# reshape y from a single row of 1,000 elements to 1,000 elements with 1 column
y = y.reshape(1000, 1)

# add our bias constant to each "row" in our array.
# np.c_ concatenates elements across the second axis, i.e. the "column" axis
# from the documentation:
#
# >>> np.c_[np.array([1,2,3]), np.array([4,5,6])]
# array([[1, 4],
#        [2, 5],
#        [3, 6]])
#
X = np.c_[X, np.ones(X.shape[0])]

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.5, random_state=42)

# random initialisation of weights - here we have one neuron
W = np.random.randn(X.shape[1], 1)
losses = []
weights = []

# training
for epoch in range(args["epochs"]):

    # preds.shape
    # (500, 1)
    preds = sigmoid_activation(train_X.dot(W))

    # error.shape
    # (500, 1)
    error = preds - train_Y

    # loss is scalar
    loss = np.sum(error**2)
    losses.append(loss)

    # calculate gradient

    # train_X.shape
    # (500, 3)
    #
    # train_X.T.shape
    # (3, 500)
    #
    # train_X.T.dot(error)
    # arrays with dimensions (3, 500) and (500, 1) = (3, 1) output
    #
    # gradient.shape
    # (3, 1)

    # gradient in this case is dot product between our features and our current
    # error.
    gradient = train_X.T.dot(error)

    # gradient descent step - move in opposite direction of gradient
    # dimensions of W are still (3, 1)
    W = W - 0.01 * gradient
    weights.append(W)
    print("epoch: {}, loss:{:.7f}".format(epoch + 1, loss))

# evaluate model
preds = predict(test_X, W)
print(classification_report(test_Y, preds))

# plots
plt.style.use("ggplot")
plt.figure()
plt.title("test data")
plt.scatter(test_X[:, 0], test_X[:, 1], marker="o", s=30)

plt.style.use("ggplot")
plt.figure()
plt.title("training loss")
plt.xlabel("epoch number")
plt.ylabel("loss")
plt.plot(np.arange(0, args["epochs"]), losses)
