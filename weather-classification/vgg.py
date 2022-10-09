import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras import Model
from keras.applications import EfficientNetB7, VGG19
from keras.applications.resnet import ResNet50
from keras.layers import Flatten, UpSampling2D, Input, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.activations import softmax, relu, sigmoid
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.callbacks import History, EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from PIL import Image


DATASET_DIR = sys.argv[1]
SAVE_DIR = sys.argv[2]
WEATHERS = ['Fog', 'Night', 'Rain', 'Snow']
EPOCH = 100
BATCH_SIZE = 32
VAL_SPLIT = 0.2


class VGG:
    def run(self):
        x_train, x_test, y_train, y_test = self.__load_dataset()
        # x_train, x_test = self.__preprocess_input(x_train=x_train, x_test=x_test)
        # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        model = self.__create_model(shape=x_train[0].shape)
        stop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=VAL_SPLIT, callbacks=[stop_callback])
        evaluation = model.evaluate(x_test, y_test, verbose=0, batch_size=BATCH_SIZE)
        self.__save_history(filepath=os.path.join(SAVE_DIR, 'history.txt'), history=history)
        self.__save_model(directory=os.path.join(SAVE_DIR), model=model)
        self.__save_evaluation(filepath=os.path.join(SAVE_DIR, 'evaluation.txt'), evaluation=evaluation)

    def __create_model(self, shape: tuple) -> Model:
        model = keras.Sequential(
            [
                VGG19(input_shape=shape, include_top=False, weights='imagenet'),
                GlobalAveragePooling2D(),
                Dense(100, activation=relu),
                Dense(100, activation=relu),
                Dense(4, activation=softmax)
            ]
        )
        model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def __load_dataset(self) -> tuple:
        count = 0
        x, y = list(), list()

        for weather in WEATHERS:
            for task in ['Test', 'Train', 'Validation']:
                directory = os.path.join(DATASET_DIR, weather, weather + '_' + task)
                for filename in os.listdir(directory):
                    image = Image.open(os.path.join(directory, filename))
                    x.append(np.array(image) / 255)
                    y.append(count)
            count += 1
        x = np.array(x)
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
        return x_train, x_test, y_train, y_test

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