import keras
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
import imageio
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
def conv2d(layer_input, filters, f_size=4):
    """Layers used during downsampling"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = InstanceNormalization()(d)
    return d


def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
    """Layers used during upsampling"""
    u = UpSampling2D(size=2)(layer_input)
    u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout_rate:
        u = Dropout(dropout_rate)(u)
    u = InstanceNormalization()(u)
    u = Concatenate()([u, skip_input])
    return u


g_AB = keras.models.load_model('C:\\Users\\ludandan\\Desktop\\Keras-GAN-master\\Keras-GAN-master\\cyclegan\\saved_model\\epoch_200\\g_AB.h5',
                               custom_objects={'conv2d': conv2d,'margin_loss':deconv2d, 'InstanceNormalization':InstanceNormalization})

im = imageio.imread('C:\\Users\\ludandan\\Desktop\\Keras-GAN-master\\Keras-GAN-master\\cyclegan\\datasets\\photo2sketch\\trainA\\00001fb010_930831.jpg').astype(np.float)
imgs = []
imgs.append(im)
imgs = np.array(imgs)/127.5 - 1.
fake = g_AB.predict(imgs)[0]
fake = 127.5*fake + 127.5
print(fake.shape)
# plt.imshow(fake)
# plt.show()


# fake = Image.fromarray(fake,'RGB')
fake = cv2.resize(fake,(224,224))
print(fake)
print(fake.shape)
# cv2.imshow('0',fake)
# cv2.waitKey(0)



