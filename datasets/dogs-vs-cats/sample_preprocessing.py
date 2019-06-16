from compviz_utils.preprocessors import ImagePreprocessor
from compviz_utils.data_loaders import DataLoader

ip = ImagePreprocessor(32, 32)
dl = DataLoader([ip])

data, labels = dl.load('./datasets/dogs-vs-cats/')