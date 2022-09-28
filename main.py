# Emily Haigh, Anthony An, Josh Leppo & Manoah Mohan

import performance
import alexNet
import tensorflow as tf
from tensorflow import keras

# dataset processing questions:
# rgb
# image dimension? input_shape = (x, y, 3)
# train with good quality?
# TODO: NEED TO UPDATE THESE WHEN WE FIGURE OUT THE DATASET STUFF
x_train = 0
y_train = 0
x_test = 0
num_classes = 4
classes = ['dark', 'rain', 'snow', 'fog']

alexNet_model = alexNet.alexNet(num_classes=4).model

alexNet_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=0.01), metrics=['accuracy'])
alexNet_model.summary()

history = alexNet_model.fit(x_train, y_train,
          epochs=50,
          #callbacks=callback  TODO: do we want any call backs?
          )

# To print the loss and accuracy graphs
performance.plot_performance(history)

alexNet_model.evaluate(x_test)

