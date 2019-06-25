from config import lisa_config
from compviz.tfannotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import cv2


def check_boxes():
    # open classes output file
    f = open(lisa_config.CLASSES_FILE, 'w')
    # this is the format that tensoflow API expects.
    # each class  has an item object associated with it which has 2 attributes:
    # id and name. id = integer representing the class and name is the human
    # readable name of class
    # ids should begin with 1 because 0 is a reserved 'background' class
    for (k, v) in lisa_config.CLASSES.items():
        item = ("item {\n"
                  "\tid: " + str(v) + "\n"      
                  "\tname: ’" + k + "’\n"
                  "}\n")
        f.write(item)
    f.close

    # initialise dictionary used to map each file name to bounding boxes
    # associated with it, then load contents of annotations file
    D = {}
    rows = open(lisa_config.ANNOT_PATH).read().strip().split('\n')
    for row in rows[1:]:
        row = row.split(",")[0].split(";")
        (imagePath, label, startX, startY, endX, endY, _) = row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))
        # if we are not interested in the label, ignore it
        if label not in lisa_config.CLASSES:
            continue

        # build path to input image then find any other bounding boxes and labels
        # associated with image.
        # these steps are important because the train test splits need to be done at
        # the image level, and not the bounding box level
        p = os.path.sep.join([lisa_config.BASE_PATH, imagePath])
        # the get method returns the value of a key, if it exists.
        # otherwise, it defaults to the default value passed into the method.
        # in this case, we default to an empty list
        b = D.get(p, [])
        # append tuple to dictionary value
        b.append((label, (startX, startY, endX, endY)))
        D[p] = b

    # create training and testing splits from our data dictionary
    (trainKeys, testKeys) = train_test_split(list(D.keys()), test_size = lisa_config.TEST_SIZE, random_state = 42)

    # initialize the data split files
    datasets = [
        ("train", trainKeys, lisa_config.TRAIN_RECORD),
        ("test", testKeys, lisa_config.TEST_RECORD)
    ]

    # build record files
    # loop over the datasets
    for (dType, keys, outputPath) in datasets:
        # initialize the TensorFlow writer and initialize the total
        # number of examples written to file
        print("[INFO] processing ’{}’...".format(dType))
        writer = tf.python_io.TFRecordWriter(outputPath)

        # loop over all the keys in the current set, where k
        # is an image path
        for k in keys:
            # load the input image from disk as a TensorFlow object
            encoded = tf.gfile.GFile(k, "rb").read()
            encoded = bytes(encoded)
            # load the image from disk again, this time as a PIL object
            # the image gets loaded in PIL/Pillow format which allows
            # for easy access to the image dimensions
            pilImage = Image.open(k)
            (w, h) = pilImage.size[:2]

            # build tfAnnot object using our class TFAnnotation()
            # parse the filename and encoding from the input path
            filename = k.split(os.path.sep)[-1]
            encoding = filename[filename.rfind(".") + 1:]
            # initialize the annotation object used to store
            # information regarding the bounding box + labels
            tfAnnot = TFAnnotation()
            tfAnnot.image = encoded
            tfAnnot.encoding = encoding
            tfAnnot.filename = filename
            tfAnnot.width = w
            tfAnnot.height = h

            # loop over bounding box info and add to tfAnnot
            # loop over the bounding boxes + labels associated with
            # the image
            for (label, (startX, startY, endX, endY)) in D[k]:
                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them
                xMin = startX / w
                xMax = endX / w
                yMin = startY / h
                yMax = endY / h

                # load the input image from disk and denormalize the
                # bounding box coordinates
                image = cv2.imread(k)
                startX = int(xMin * w)
                startY = int(yMin * h)
                endX = int(xMax * w)
                endY = int(yMax * h)

                # draw the bounding box on the image
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

                print('label is ' + label)
                cv2.imshow("Image", image)
                cv2.waitKey(0)





