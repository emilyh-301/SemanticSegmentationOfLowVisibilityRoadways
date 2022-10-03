import os

input_dir = '/home/nughufer/COMP597_Files/Rain_Train/'
target_dir = '/home/nughufer/COMP597_Files/Rain_Train_An/'
val_input_dir = '/home/nughufer/COMP597_Files/Rain_Val/'
val_target_dir = '/home/nughufer/COMP597_Files/Rain_Val_An/'
img_size = (128,128)
num_classes = 19
batch_size = 32

train_input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith('.png')
    ]
)
train_target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)

val_input_img_paths = sorted(
    [
        os.path.join(val_input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith('.png')
    ]
)
val_target_img_paths = sorted(
    [
        os.path.join(val_target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class RoadImages(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i:i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i:i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='uint8')
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size)
            # y[j] = np.expand_dims(img, 2)
            y[j] = img
            y[j] -= 1
        return x, y

unet = uNet()
model = unet.get_Model()

train_gen = RoadImages(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)

val_gen = RoadImages(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')

callbacks = [
    keras.callbacks.ModelCheckpoint('road_segmentation.h5',save_best_only=True)
]

epochs = 1
model.fit(train_gen,epochs=epochs,validation_data=val_gen,callbacks=callbacks)
