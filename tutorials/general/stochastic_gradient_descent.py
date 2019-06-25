
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


def next_batch(X, y, batch_size):
    for i in np.arange(0, X.shape[0], batch_size):
        yield X[i: i + batch_size], y[i: i + batch_size]

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
ap.add_argument('-a', '--alpha', type=float, default=0.01, help='learning rate')
ap.add_argument('-b', '--batch-size', type=int, default=32, help='size of SGD mini-batches')
args = vars(ap.parse_args())

# the following sample size with a batch size of 32 results in the
# last batch having only 8 observations
X, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape(y.shape[0], 1)

# bias trick
X = np.c_[X, np.ones(X.shape[0])]
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.5, random_state=42)

W = np.random.randn(X.shape[1], 1)
losses = []

for epoch in range(args['epochs']):
    epoch_loss = []
    for batch_X, batch_y in next_batch(X, y, args['batch_size']):
        preds = sigmoid_activation(batch_X.dot(W))
        error = preds - batch_y
        epoch_loss.append(np.sum(error ** 2))
        gradient = batch_X.T.dot(error)

        # gradient descent step takes place inside the batch loop
        W += -args['alpha'] * gradient

    # take the average mini-batch loss for the current epoch.
    # this becomes the loss for this epoch
    loss = np.average(epoch_loss)
    losses.append(loss)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("epoch={}, loss={:.7f}".format(epoch + 1, loss))

print("evaluating...")
preds = predict(test_x, W)
print(classification_report(test_y, preds))

plt.style.use('ggplot')
plt.figure()
plt.title("test data - mini-batch stochastic gradient descent")
plt.scatter(test_x[:, 0], test_x[:, 1], marker="o", s=30)

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(args['epochs']), losses)
plt.title('training loss - mini-batch stochastic gradient descent')
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.show()
