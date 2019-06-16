# knn on raw pixel arrays
#
# usage example:
#   - cd into computer-vision root folder
#   - in terminal:
#
#       python image_recognition/knn.py -d datasets/dogs-vs-cats
#

import sys
import os
utils_path = os.path.expanduser('~/github/computer-vision/')
sys.path.append(utils_path)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from compviz_utils.preprocessors import ImagePreprocessor
from compviz_utils.data_loaders import DataLoader
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbours", type=int, default=1,
                help="number of nearest neighbours for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
                help="number of jobs for k-NN distance (-1 uses all available cores")
args = vars(ap.parse_args())

print("loading images...")

# create preprocessors - resize images to 32x32 pixels
preproc_resize = ImagePreprocessor(32, 32)

# load and process
data_loader = DataLoader(preprocessors=[preproc_resize])
images, labels = data_loader.load('./datasets/dogs-vs-cats/')

# flatten array into column vector
# first axis (==0) is number of examples
# second axis (==1) is elements in column vector
images_flattened = images.reshape((images.shape[0], 32*32*3))

# encode target labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# train vs test split
x_train, x_test, y_train, y_test = train_test_split(images_flattened, labels, test_size=0.25, random_state=42)

# train model
model = KNeighborsClassifier(n_neighbors=args['neighbours'], n_jobs=args['jobs'])
model.fit(x_train, y_train)

# evaluate
results_report = classification_report(y_test, model.predict(x_test), target_names=le.classes_)
print(results_report)
