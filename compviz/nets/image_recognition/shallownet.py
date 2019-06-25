
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):

        # width, height and depth are properties of images we are using to train this net
        # classes is the number of classes we are trying to predict
        model = Sequential()

        # channels last
        input_shape = height, width, depth

        if K.image_data_format() == 'channels_first':
            input_shape = depth, height, width

        # convolutional layer with 32 filters of 3x3
        # "same convolution" applied via padding='same'. input dimensions and output of convolution
        # will be the same.
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
