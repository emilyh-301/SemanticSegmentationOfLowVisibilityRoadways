import os
import sys
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


if len(sys.argv) < 3:
    print('You should set a dataset path and model path.')
    sys.exit(1)

DATASET_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]


def load_dataset(path: str) -> list:
    # Load an image
    if os.path.isfile(path=path):
        image = Image.open(fp=path)
        image = np.asarray(image) / 255.0
        image = np.reshape(image, (1, *image.shape))
        return [image]

    # Load multiple images in a directory
    else:
        image_arr = list()
        filenames = os.listdir(path)
        for filename in filenames:
            image = Image.open(fp=os.path.join(path, filename))
            image = np.asarray(image) / 255.0
            image = np.reshape(image, (1, *image.shape))
            image_arr.append(image)
        return image_arr


dataset = load_dataset(path=DATASET_PATH)
cls_model = load_model(filepath=os.path.join(MODEL_PATH, 'Classification'))
fog_model = load_model(filepath=os.path.join(MODEL_PATH, 'Fog'))
Night_model = load_model(filepath=os.path.join(MODEL_PATH, 'Night'))
Rain_model = load_model(filepath=os.path.join(MODEL_PATH, 'Rain'))
Snow_model = load_model(filepath=os.path.join(MODEL_PATH, 'Snow'))

for data in dataset:
    predictions = cls_model.predict(x=data)
    pred_cls = np.argmax(predictions, axis=1)[0]

    # Fog
    if pred_cls == 0:
        pass

    # Night
    elif pred_cls == 1:
        pass

    # Rain
    elif pred_cls == 2:
        pass

    # Snow
    elif pred_cls == 3:
        pass