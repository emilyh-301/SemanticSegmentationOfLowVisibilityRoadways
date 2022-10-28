from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard

from data_generator import DataGenerator
from models.UNet.UNet import UNet
from predict import Predict

dataset_path = "/home/nughufer/COMP597_Files/"

validation_rgb = list(Path(dataset_path + 'Rain_Val').glob('*.png'))
validation_annotated = list(Path(dataset_path + 'Rain_Val_An').glob('*_gt_labelColor.png'))

# Splits training and testing data then gets corresponding files based on similar file names
training_rgb, test_rgb = train_test_split(list(Path(dataset_path + 'Rain_Train').glob('*.png')), test_size=0.1)
training_annotated = [y for x in training_rgb for y in
                      list(Path(dataset_path + 'Rain_Train_An').glob('*_gt_labelColor.png'))
                      if x.name.strip('_rgb_anon.png') == y.name.strip('_gt_labelColor.png')]
test_annotated = [y for x in test_rgb for y in list(Path(dataset_path + 'Rain_Train_An').glob('*_gt_labelColor.png'))
                  if x.name.strip('_rgb_anon.png') == y.name.strip('_gt_labelColor.png')]

# [(image, annotated image)]
training_duo = [(x, y) for x, y in zip(training_rgb, training_annotated)]
testing_duo = [(x, y) for x, y in zip(test_rgb, test_annotated)]
validation_duo = [(x, y) for x, y in zip(validation_rgb, validation_annotated)]

image_size = (256, 256)
classes_df = pd.read_csv("class_dict.csv")
classes = []
for index, item in classes_df.iterrows():
    classes.append(np.array([item['r'], item['g'], item['b']]))
num_classes = len(classes)

model = UNet(image_size, 3, 64, num_classes).model
model.compile(optimizer='adam', loss='categorical_crossentropy' ,metrics=['accuracy'])
# To load previously trained model
# model.load_weights('./fog-weights.h5')

my_callbacks = [
    CSVLogger("./models/UNet/logs/log.csv", separator=",", append=False),
    ModelCheckpoint(filepath='./models/UNet/fog-weights.h5', save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True),
    TensorBoard(log_dir='./models/UNet/logs')
]

training_data = DataGenerator(training_duo+testing_duo,classes,num_classes,batch_size=3, dim=image_size ,shuffle=True)
training_steps = training_data.__len__()
validation_data = DataGenerator(validation_duo,classes,num_classes,batch_size=3, dim=image_size ,shuffle=True)
validation_steps = validation_data.__len__()

model_train = model.fit(training_data, epochs=1, callbacks=my_callbacks, validation_data=validation_data, steps_per_epoch=training_steps, validation_steps=validation_steps)
prediction = Predict(image_size, model, classes)

for x in range(0, 10):
    prediction.predict(validation_duo[x])






