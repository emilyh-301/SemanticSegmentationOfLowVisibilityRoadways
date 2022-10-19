import os
import pandas as pd
import numpy as np
import tensorflow
from tensorflow import keras
from keras import Model
from keras.layers import Flatten, UpSampling2D, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, BatchNormalization, Dropout, Conv2D, MaxPool2D
from keras.activations import softmax, relu, sigmoid
from keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from keras.callbacks import History, EarlyStopping
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image


WEATHERS = ['Fog', 'Night', 'Rain', 'Snow']
TASKS = ['Test', 'Train', 'Validation']
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2
RANDOM_STATE = 42
EPOCH = 100
PATIENCE = 10


class VGG:
    def __init__(self, dataset_dir: str, save_dir: str, unit_count=100, lr=0.001, batch_size=32):
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.fc_unit_count = unit_count
        self.lr = lr
        self.batch_size = batch_size

    def run(self) -> None:
        x_train, x_test, y_train, y_test = self.__load_dataset()
        model = self.__create_model(shape=x_train[0].shape)
        stop_callback = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        history = model.fit(x_train, y_train, epochs=EPOCH, batch_size=self.batch_size, validation_split=VAL_SPLIT, callbacks=[stop_callback])
        evaluation = model.evaluate(x_test, y_test, verbose=0, batch_size=self.batch_size)
        self.__save_history(filepath=os.path.join(self.save_dir, 'history.txt'), history=history)
        self.__save_model(directory=os.path.join(self.save_dir), model=model)
        self.__save_evaluation(filepath=os.path.join(self.save_dir, 'evaluation.txt'), evaluation=evaluation)

    def __create_model(self, shape: tuple) -> Model:
        model = keras.Sequential(
            [
                VGG19(input_shape=shape, include_top=False, weights='imagenet'),
                GlobalAveragePooling2D(),
                # GlobalMaxPooling2D(),
                Dense(self.fc_unit_count, activation=relu),
                Dense(self.fc_unit_count, activation=relu),
                Dense(4, activation=softmax)
            ]
        )
        model.compile(optimizer=SGD(learning_rate=self.lr), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        return model

    def __load_dataset(self) -> tuple:
        count = 0
        x_train, x_test, y_train, y_test = None, None, None, None

        for weather in WEATHERS:
            x, y = list(), list()
            for task in TASKS:
                directory = os.path.join(self.dataset_dir, weather, weather + '_' + task)
                for filename in os.listdir(directory):
                    image = Image.open(os.path.join(directory, filename))
                    x.append(np.array(image) / 255)
                    y.append(count)
            count += 1

            split_data = train_test_split(np.array(x), np.array(y), test_size=TEST_SPLIT, random_state=RANDOM_STATE, shuffle=True)
            if x_train is None:
                x_train, x_test, y_train, y_test = split_data
            else:
                x_train = np.append(x_train, split_data[0], 0)
                x_test = np.append(x_test, split_data[1], 0)
                y_train = np.append(y_train, split_data[2], 0)
                y_test = np.append(y_test, split_data[3], 0)

        x_train, y_train = shuffle(x_train, y_train, random_state=RANDOM_STATE)
        x_test, y_test = shuffle(x_test, y_test, random_state=RANDOM_STATE)

        return x_train, x_test, y_train, y_test

    def __save_history(self, filepath: str, history: History):
        with open(filepath, mode='w') as f:
            pd.DataFrame(history.history).to_json(f)

    def __save_model(self, directory: str, model: Model) -> None:
        model.save(directory)

    def __save_evaluation(self, filepath: str, evaluation: tuple) -> None:
        content = 'Test loss: ' + str(evaluation[0]) + '\n' + 'Test accuracy: ' + str(evaluation[1])
        with open(filepath, mode='w') as f:
            f.write(content)