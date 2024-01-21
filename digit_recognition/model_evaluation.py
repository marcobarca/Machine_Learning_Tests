import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading the dataset
mnist = tf.keras.datasets.mnist

# Splitting the dataset into training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.load_model('digit_recognition_3dense.model')

# Evaluating the model
# Loss is sparse_categorical_crossentropy (loss function)
#   - It is used when the classes are mutually exclusive (each entry is exactly one class)

loss, accuracy = model.evaluate(x_test, y_test)

print(loss) # 55.68372344970703
print(accuracy) # 0.9570000171661377
