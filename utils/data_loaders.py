import numpy as np
import cv2
import os

# preprocessors should be a list of preprocessor objects
# so that different preprocessors can be applied to each image
# sequentially


class DataLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors

        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths):
        data = []
        labels = []

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)

            # get label and apply processors sequentially
            # assumes that folder of the image contains class name.
            label = image_path.split(os.path.sep)[-2]
            image = self._apply_processors(image)

            data.append(image)
            data.append(label)

            self._print_progress(i, image_paths)

        return np.array(data), np.array(labels)

    def _apply_processors(self, image):
        for p in self.preprocessors:
            image = p.preprocess(image)
        return image

    @staticmethod
    def _print_progress(i, image_paths):
        if i % 10:
            print("processed {}/{} images".format(i + 1, len(image_paths)))

