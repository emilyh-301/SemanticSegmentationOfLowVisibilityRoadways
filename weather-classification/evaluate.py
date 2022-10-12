import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


DATASET_DIR = sys.argv[1]
MODEL_DIR = sys.argv[2]
WEATHERS = ['Fog', 'Night', 'Rain', 'Snow']
TASKS = ['Test', 'Train', 'Validation']
BATCH_SIZE = 32


def load_dataset() -> tuple:
    count = 0
    x, y = list(), list()

    for weather in WEATHERS:
        for task in TASKS:
            directory = os.path.join(DATASET_DIR, weather, weather + '_' + task)
            for filename in os.listdir(directory):
                image = Image.open(os.path.join(directory, filename))
                x.append(np.array(image) / 255)
                y.append(count)
        count += 1
    return np.array(x), np.array(y)

x, y = load_dataset()
model = load_model(MODEL_DIR)
evaluation = model.evaluate(x, y, verbose=0, batch_size=BATCH_SIZE)
print(evaluation)