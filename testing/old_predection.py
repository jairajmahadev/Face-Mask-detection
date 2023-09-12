import cv2
import tensorflow as tf
import time

CATEGORIES = ["with_mask", "without_mask"]
image = 'yesmask.jpg'
IMG_SIZE = 50
img_array = cv2.imread(image, cv2.COLOR_BGR2RGB)
new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))

#cv2.imshow("Frame", new_array)

def prepare(filepath):
  IMG_SIZE = 224
  img_array = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
  new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
  return new_array.reshape(-1, IMG_SIZE,IMG_SIZE, 3)

model = tf.keras.models.load_model("old.model")

final_image = prepare(image)

start_time = time.time()
predection = model.predict([final_image])
end_time = time.time()

final_time = end_time - start_time

print("Prediction is: ")
print(CATEGORIES[int(predection[0][0])])
print("Time taken to process: ")
print(final_time)
