import os

# location of raw data
BASE_PATH = 'lisa'

ANNOT_PATH = os.path.sep.join([BASE_PATH, 'allAnnotations.csv'])

TRAIN_RECORD = os.path.sep.join([BASE_PATH, 'records\\training.record'])
TEST_RECORD = os.path.sep.join([BASE_PATH, 'records\\testing.record'])
CLASSES_FILE = os.path.sep.join([BASE_PATH, 'records\\classes.pbtxt'])

# 75% training, 25% testing
TEST_SIZE = 0.25

# we are only training to detect 3 of the 47 traffic signs
CLASSES = {'pedestrianCrossing' : 1, 'signalAhead' : 2, 'stop' : 3}