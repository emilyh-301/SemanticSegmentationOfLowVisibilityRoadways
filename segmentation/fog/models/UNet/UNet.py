from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Activation, Dropout


class UNet:
    def __init__(self, dimensions, channels, filters, n_classes):
        input_layer = Input(shape=(dimensions + (channels,)), name='input')

        conv_1_layer = self.__gen_convolution(input_layer, filters)
        conv_1_pooling = MaxPooling2D(pool_size=(2, 2))(conv_1_layer)

        conv_2_layer = self.__gen_convolution(conv_1_pooling, filters * 2)
        conv_2_pooling = MaxPooling2D(pool_size=(2, 2))(conv_2_layer)

        conv_3_layer = self.__gen_convolution(conv_2_pooling, filters * 4)
        conv_3_pooling = MaxPooling2D(pool_size=(2, 2))(conv_3_layer)

        conv_4_layer = self.__gen_convolution(conv_3_pooling, filters * 8)
        conv_4_pooling = MaxPooling2D(pool_size=(2, 2))(conv_4_layer)
        conv_4_dropout = Dropout(0.5)(conv_4_pooling)

        conv_5_layer = self.__gen_convolution(conv_4_dropout, filters * 16)
        conv_5_dropout = Dropout(0.5)(conv_5_layer)

        deconv_6_layer = self.__gen_deconvolution(conv_5_dropout, filters * 8, conv_4_layer)
        deconv_6_dropout = Dropout(0.5)(deconv_6_layer)

        deconv_7_layer = self.__gen_deconvolution(deconv_6_dropout, filters * 4, conv_3_layer)
        deconv_7_dropout = Dropout(0.5)(deconv_7_layer)

        deconv_8_layer = self.__gen_deconvolution(deconv_7_dropout, filters * 2, conv_2_layer)
        deconv_9_layer = self.__gen_deconvolution(deconv_8_layer, filters * 2, conv_1_layer)
        output_layer = Conv2D(filters=n_classes, kernel_size=(1, 1), activation='softmax')(deconv_9_layer)

        self.model = Model(inputs=input_layer, outputs=output_layer, name='UNet')

    def __gen_convolution(self, prev, filters):
        temp = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                         activation='relu')(prev)
        temp = BatchNormalization()(temp)
        temp = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                         activation='relu')(temp)
        return BatchNormalization()(temp)

    def __gen_deconvolution(self, prev, filters, residual):
        temp = Conv2DTranspose(filters=filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(prev)
        temp = concatenate([temp, residual], axis=3)
        return self.__gen_convolution(temp, filters)