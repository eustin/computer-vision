
import cv2
from keras.preprocessing.image import img_to_array


class ImagePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def process(self, image):
        resized = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        return resized


class ImageToArray:
    def __init__(self, data_format=None):
        # data format here is either None, 'channels_first' or 'channels_last'
        # if None, default setting of image_data_format in ~/.keras/keras.json will be used.
        self.data_format = data_format

    def preprocess(self, image):
        return img_to_array(image, data_format=self.data_format)


