import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time


# dataset processing questions:
# rgb
# image dimension? input_shape = (x, y, 3)
# train with good quality?

# NEED TO UPDATE THESE WHEN WE FIGURE OUT THE DATASET STUFF
train_ds = 0
validation_ds = 0
test_ds = 0
num_classes = 0

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.01), metrics=['accuracy'])
model.summary()

history = model.fit(train_ds,
          epochs=50,
          validation_data=validation_ds,
          validation_freq=1,
          #callbacks=[tensorboard_cb]  do we want any call backs
          )

# Open up a terminal at the directory level where the TensorBoard log folder exists and run the following command:
# tensorboard --logdir logs
# this shows loss and val loss graph, also accuracy graph

model.evaluate(test_ds)