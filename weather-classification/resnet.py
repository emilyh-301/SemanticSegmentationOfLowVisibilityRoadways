import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras import Model
from keras.applications.resnet import ResNet50
from keras.layers import Flatten, UpSampling2D, Input, GlobalAveragePooling2D, Dense
from keras.activations import softmax, relu
from keras.optimizers import SGD, Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import History, EarlyStopping

DATASET_DIR = sys.argv[1]
MODEL_DIR = ''
HISTORY_FILEPATH = ''
EVAL_FILEPATH = ''
WEATHERS = ['Fog', 'Night', 'Rain', 'Snow']
EPOCH = 10
BATCH_SIZE = 128
VAL_SPLIT = 0.1


class ResNet:
    def run(self):
        x_train, y_train, x_test, y_test = self.__load_dataset()
        # x_train, x_test = self.__preprocess_input(x_train=x_train, x_test=x_test)
        model = self.__create_model(shape=x_train[0].shape)
        history = model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT)
        evaluation = model.evaluate(x_test, y_test, verbose=0, batch_size=BATCH_SIZE)
        self.__save_history(filepath=HISTORY_FILEPATH, history=history)
        self.__save_model(directory=MODEL_DIR, model=model)
        self.__save_evaluation(filepath=EVAL_FILEPATH, evaluation=evaluation)

    def __create_model(self, shape: tuple) -> Model:
        layer_input = Input(shape=shape)
        # upsampling = UpSampling2D(size=(7, 7))(layer_input)
        resnet = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')(layer_input)
        resnet.trainable = False
        # output_layers = GlobalAveragePooling2D()(resnet)
        # output_layers = Flatten()(output_layers)
        layer_output = Dense(100, activation=relu)(resnet)
        layer_output = Dense(100, activation=relu)(layer_output)
        layer_output = Dense(4, activation=softmax)(layer_output)
        model = Model(inputs=layer_input, outputs=layer_output)
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        # model.summary()
        return model

    def __load_dataset(self) -> tuple:
        pass

    def __preprocess_input(self, x_train: np.ndarray, x_test: np.ndarray) -> tuple:
        x_train = keras.applications.resnet50.preprocess_input(x_train.astype('float32'))
        x_test = keras.applications.resnet50.preprocess_input(x_test.astype('float32'))
        return x_train, x_test

    def __save_history(self, filepath: str, history: History):
        with open(filepath, mode='w') as f:
            pd.DataFrame(history.history).to_json(f)

    def __save_model(self, directory: str, model: Model) -> None:
        model.save(directory)

    def __save_evaluation(self, filepath: str, evaluation: tuple) -> None:
        content = 'Test loss: ' + str(evaluation[0]) + '\n' + 'Test accuracy: ' + str(evaluation[1])
        with open(filepath, mode='w') as f:
            f.write(content)
