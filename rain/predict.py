from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from matplotlib import pyplot as plt

class Predict:
    def __init__(self, image_size, model, classes):
        self.image_size = image_size
        self.model = model
        self.classes = classes

    # image_path_pair = (Image Path, Annotated Path)
    def predict(self, image_path_pair):
        image = img_to_array(load_img(image_path_pair[0] , target_size=self.image_size))
        annotated_image = img_to_array(load_img(image_path_pair[1], target_size=self.image_size))

        image_expanded = np.expand_dims(img_to_array(load_img(image_path_pair[0], target_size=self.image_size)) / 255., axis=0)
        prediction = np.argmax(self.model.predict(image_expanded)[0], axis=2)
        prediction_colored = np.array(self.classes)[prediction].astype(np.uint8)
        self.__plot(image, annotated_image, prediction_colored)

    def __plot(self, image, annotated_image, prediction):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.set_figheight(20)
        fig.set_figwidth(20)
        ax1.set_title('Image')
        ax1.imshow(image / 255.)
        ax2.set_title('Annotated Image')
        ax2.imshow(annotated_image / 255.)
        ax3.set_title('Prediction')
        ax3.imshow(prediction / 255.)
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')