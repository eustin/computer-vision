
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from compviz.preprocessors import ImagePreprocessor
from compviz.data_loaders import DataLoader
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
args = vars(ap.parse_args())

# preprocess data ------------------------------------------------------------------------------------------------------
ip = ImagePreprocessor(32, 32)
dl = DataLoader(preprocessors=[ip])
data, labels = dl.load(args['dataset'])

# stack images into column vectors
data = data.reshape(data.shape[0], 32*32*3)

# convert labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# train test split done on images converted into column vectors
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=5)

# try "no regularisation", "L1" and "L2" regularisation
for r in (None, "l1", "l2"):
    print("training model with '{}' penalty".format(r))
    model = SGDClassifier(loss='log', penalty=r, max_iter=1000, learning_rate='constant', eta0=0.01, random_state=42)
    model.fit(train_x, train_y)

    print("evaluating model...")
    acc = model.score(test_x, test_y)
    print("'{}' penalty accuracy is {:.2f}%\n".format(r, acc*100))
