from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils.preprocessors import ImagePreprocessor
from utils.data_loaders import DataLoader
from imutils import paths
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
images_flattened = images.reshape((images.shape[0], 3072))
