import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

model = tf.keras.models.load_model('digit_recognition.model')

image_number = 0
while os.path.isfile(f"./digits/{image_number}.jpg"):
    try:
        # [:,:,0] is used to convert the image to grayscale
        img = cv2.imread(f"./digits/{image_number}.jpg")[:,:,0]
        # Resizing the image to 28x28
        img = cv2.resize(img, (28, 28))
        # Inverting the image (black to white and vice versa)
        img = np.invert(np.array([img]))
        # Predicting the number
        prediction = model.predict(img)
        print(f"The number is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("An error occured")
    image_number += 1
