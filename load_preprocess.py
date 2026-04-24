import tensorflow as tf

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


import numpy as np

# Normalize images (-1 to 1)
x_train = (x_train.astype('float32') - 127.5) / 127.5

# Flatten labels
y_train = y_train.flatten()

num_classes = 10
