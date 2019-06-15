
import cv2


class ImagePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

    def process(self, image):
        resized = cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        return resized


