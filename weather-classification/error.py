import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle
from PIL import Image


DATASET_DIR = sys.argv[1]
MODEL_PATH = sys.argv[2]
WEATHERS = ['Fog', 'Night', 'Rain', 'Snow']
TASKS = ['Test', 'Train', 'Validation']
BATCH_SIZE = 16


def load_dataset() -> tuple:
    count = 0
    x, y, filenames = list(), list(), list()

    for weather in WEATHERS:
        for task in TASKS:
            directory = os.path.join(DATASET_DIR, weather, weather + '_' + task)
            for filename in os.listdir(directory):
                image = Image.open(os.path.join(directory, filename))
                x.append(np.array(image) / 255)
                y.append(count)
                filenames.append(filename)
        count += 1
    x, y, filenames = np.array(x), np.array(y), np.array(filenames)
    x, y, filenames = shuffle(x, y, filenames, random_state=42)
    return x, y, filenames

x, y, filenames = load_dataset()
model = load_model(MODEL_PATH)
model.summary()

predictions = model.predict(x, verbose=0, batch_size=BATCH_SIZE)
predictions = np.argmax(predictions, axis=1)
for i, pred in enumerate(predictions):
    if pred != y[i]:
        print(pred, y[i], filenames[i])