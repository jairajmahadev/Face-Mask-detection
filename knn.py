from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import cv2
import os

def createImageFeatures(image, size=(32, 32)):
    # resize the image
    image = cv2.resize(image, size)
    # flatten the image
    pixel_list = image.flatten()
    return pixel_list

print("Reading all images")
image_paths = list(paths.list_images("dataset"))
raw_images = []
labels = []

# loop over the input images
for (i, image_path) in enumerate(image_paths):
    image = cv2.imread(image_path)
    label = image_path.split(os.path.sep)[-1].split(".")[0]
    # extract raw pixel intensity "features
    pixels = createImageFeatures(image)
    raw_images.append(pixels)
    labels.append(label)

raw_images = np.array(raw_images)
labels = np.array(labels)

(train_X, test_X, train_y, test_y) = train_test_split(
    raw_images, labels, test_size=0.25, random_state=0)
    
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_X, train_y)
acc = model.score(test_X, test_y)
print("Raw pixel accuracy: {:.2f}%".format(acc * 100))

image = cv2.imread("dog.jpg")
dog = createImageFeatures(image)
dog = np.array([dog])
print(model.predict(dog))