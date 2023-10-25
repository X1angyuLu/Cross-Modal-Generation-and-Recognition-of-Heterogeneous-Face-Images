import keras
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Lambda, Activation, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.python.keras.utils.data_utils import get_file
from keras_vggface import utils
import imageio
from keras_vggface.vggface import VGGFace
from keras import backend as K
from keras.models import Model
from keras.optimizers import adam_v2
Adam = adam_v2.Adam(learning_rate=0.0002)
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import cv2
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


def get_syn_model():
    custom_objects={'conv2d': conv2d,'margin_loss':deconv2d, 'InstanceNormalization':InstanceNormalization}
    g_AB = keras.models.load_model('C:\\Users\\ludandan\\Desktop\\Keras-GAN-master\\Keras-GAN-master\\cyclegan\\saved_model\\epoch_200\\g_AB.h5',custom_objects = custom_objects)
    g_BA = keras.models.load_model('C:\\Users\\ludandan\\Desktop\\Keras-GAN-master\\Keras-GAN-master\\cyclegan\\saved_model\\epoch_200\\g_BA.h5', custom_objects = custom_objects)
    return g_AB, g_BA
