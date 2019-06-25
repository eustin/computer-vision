
import numpy as np
import cv2

np.random.seed(1)

labels = ['dog', 'cat']

# read image
image = cv2.imread('./datasets/dogs-vs-cats/dog/dog.10.jpg')
cv2.imshow('Doggo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# initialise weights matrix and bias vector
W = np.random.randn(2, 3072)
b = np.random.randn(2)

# flatten image into column vector and calculate scores
# for each class
image_flattened = cv2.resize(image, (32, 32)).flatten()
scores = W.dot(image_flattened) + b

# evaluate
for label, score in zip(labels, scores):
    print('{}: {}'.format(label, score))

max_score_idx = np.argmax(scores)
predicted_class = labels[max_score_idx]
class_score = scores[max_score_idx]
print("Predicted class is '{}' with score {}".format(predicted_class, class_score))

cv2.putText(image, "Label: {}".format(predicted_class), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2)
cv2.imshow('Doggo', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
