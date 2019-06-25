from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from compviz.preprocessors import ImagePreprocessor, ImageToArray
from compviz.data_loaders import DataLoader
from compviz.nets.image_recognition.shallownet import ShallowNet
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

plt.style.use('ggplot')

image_dataset_path = './datasets/dogs-vs-cats/'

image_processor = ImagePreprocessor(32, 32)
image_to_array = ImageToArray()
data_loader = DataLoader(preprocessors=[image_processor, image_to_array])
data, labels = data_loader.load(image_dataset_path)

# scale pixel values
data = data.astype('float') / 255.0

# train test split
train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25, random_state=42)

# encode classes as integers and take note of mapping from label name to integer
# in this case, 'cat' == 0, 'dog' == 1
le = LabelEncoder()
le.fit(train_y)

label_names = le.classes_
print(label_names)

test_y = le.transform(test_y)
train_y = le.transform(train_y)
train_y = to_categorical(train_y, num_classes=2)
test_y = to_categorical(test_y, num_classes=2)

optimiser = SGD(lr=0.005)
model = ShallowNet.build(32, 32, 3, classes=2)
model.summary()
model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])

history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, epochs=100, verbose=1)

# predict
predictions = model.predict(test_x, batch_size=32)
predicted_classes = predictions.argmax(axis=1)

# evaluate
print(classification_report(test_y.argmax(axis=1), predicted_classes, target_names=label_names))

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()