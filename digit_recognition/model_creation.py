import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading the dataset
mnist = tf.keras.datasets.mnist

# Splitting the dataset into training and testing data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing the data (scaling the data between 0 and 1)
# axis=1 means that we are normalizing the data row-wise
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Building the model (Sequential is the simplest model in keras)
model = tf.keras.models.Sequential()

# Flattening the input layer
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

# Adding dense layers (128 neurons in the first layer and 128 neurons in the second layer)
# Activation function is relu (rectified linear - sigmoid function)
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Adding the output layer (10 neurons because we have numbers fronm 0 to 9)
# Activation function is softmax (probability distribution - sum of all the values is 1)
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Compiling the model
#   * Optimizer is adam (stochastic gradient descent)
#       - It is used for faster optimization
#   * Loss is sparse_categorical_crossentropy (loss function)
#       - It is used when the classes are mutually exclusive (each entry is exactly one class)
#   * Metrics is accuracy (accuracy of the model)
#       - It is used to monitor the training and testing steps
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, epochs=3)

#Saving the model
model.save('digit_recognition_3dense.model')


