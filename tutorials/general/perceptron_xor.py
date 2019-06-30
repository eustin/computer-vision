#
# the perceptron can't learn the XOR function
#

import numpy as np
from tutorials.general.perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# define XOR ground truth labels - XOR is true iff one is TRUE and the other is FALSE
y = np.array([[0], [1], [1], [0]])

p = Perceptron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# test to see if we have learnt the 'xor' function
for x, target in zip(X, y):
    pred = p.predict(x)
    print("[INFO] data={}, ground-truth={}, pred={}".format(x, target[0], pred))
