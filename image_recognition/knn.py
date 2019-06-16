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

