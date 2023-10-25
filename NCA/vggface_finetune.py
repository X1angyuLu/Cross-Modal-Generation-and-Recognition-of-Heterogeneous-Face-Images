import keras
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Lambda, Activation, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
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
import ft_data_loader
from keras_vggface.utils import preprocess_input
from keras.preprocessing import image
import numpy as np
import cv2

def VggFace():
    model = VGGFace(model='vgg16', include_top=True, input_shape=(224, 224, 3), pooling='avg')
    model = Model(inputs = model.input, outputs = model.get_layer('fc7').output)
    # output = model.output
    # norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(output)
    #
    # vgg = Model(inputs=model.input, outputs=norm)
    return model

def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 0.1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)


def GetModel():
    embedding_model = VggFace()
    input_shape = (224, 224, 3)
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]

    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    return embedding_model, triplet_model

if __name__=='__main__':
    embedding_model, triplet_model = GetModel()
    triplet_model.load_weights('C:\\Users\\ludandan\\Desktop\\VGG16\\weights\\')
    embedding_model = triplet_model.get_layer('vggface_vgg16')
    for layer in embedding_model.layers[-2:]:
        layer.trainable = True
    for layer in embedding_model.layers[:-2]:
        layer.trainable = False
    triplet_model.compile(loss=None, optimizer=Adam)
    generator_train = ft_data_loader.TripletGenerator(domain='A', begin=0, end=1194) #domain='A' means recognize a sketch in the photo domain
    #generator_valid = data_loader.TripletGenerator(domain='B', begin=250, end=1194)
    save_fun = tf.keras.callbacks.ModelCheckpoint(filepath='C:\\Users\\ludandan\\Desktop\\VGG16\\weights\\',
                                                  monitor='loss',
                                                  verbose=1,
                                                  save_best_only=True,
                                                  save_weights_only=True,
                                                  mode='min')
    history = triplet_model.fit_generator(generator_train,
                                          #validation_data=generator_valid,
                                          #validation_steps=500,
                                          epochs=400,
                                          verbose=1,
                                          workers=1,
                                          steps_per_epoch=200,
                                          callbacks=[save_fun])

    embedding_model.save_weights('C:\\Users\\ludandan\\Desktop\\VGG16\\weights\\embedding\\')
    #/usr/directory/weights.{epoch:02d}-{val_loss:.2f}.hdf5