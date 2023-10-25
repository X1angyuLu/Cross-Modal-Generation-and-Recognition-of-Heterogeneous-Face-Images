from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply, Lambda
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_vggface.vggface import VGGFace
from keras import backend as K
from keras.models import Model
from keras import Sequential, Model
import numpy as np
import tensorflow as tf
from keras.optimizers import adam_v2
Adam = adam_v2.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999)

class vicreg():
    def __init__(self, input_shape, output_shape):
        self.model = self.build_transformer(input_shape, output_shape)
        self.optimizer = Adam


    def build_transformer(self, input_shape, output_shape):
        d0 = Input(shape=(1, input_shape))
        d1 = Dense(output_shape, activation='relu')(d0)
        d2 = Dense(output_shape, activation='relu')(d1)
        d3 = Lambda(lambda x: K.l2_normalize(x, axis=1))(d2)
        model = Model(d0, d3)
        model.summary()
        return model

    def standardize_columns(self, x):
        col_mean = tf.math.reduce_mean(x, axis=0)
        norm_col = x - col_mean
        return norm_col

    def off_diagonal(self, x):
        n = tf.shape(x)[0]
        flattened = tf.reshape(x, [-1])[:-1]
        off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
        off_diag = tf.reshape(off_diagonals, [-1])
        return off_diag

    def loss(self, za, zb):
        # compute the diagonal
        batch_size = tf.shape(za)[0]

        # distance loss to measure similarity between representations
        sim_loss = tf.keras.losses.MeanSquaredError()(za, zb)

        za = self.standardize_columns(za)
        zb = self.standardize_columns(zb)

        # std loss to maximize variance(information)
        std_za = tf.sqrt(tf.math.reduce_variance(za, 0) + 1e-04)
        std_zb = tf.sqrt(tf.math.reduce_variance(zb, 0) + 1e-04)
        std_loss_za = tf.reduce_mean(tf.max(0, 1 - std_za))
        std_loss_zb = tf.reduce_mean(tf.max(0, 1 - std_zb))
        std_loss = std_loss_za / 2 + std_loss_zb / 2

        # cross-correlation matrix axa
        ca = tf.matmul(za, za, transpose_a=True)
        ca = ca / tf.cast(batch_size - 1, dtype="float32")

        # cross-correlation matrix bxb
        cb = tf.matmul(zb, zb, transpose_a=True)
        cb = cb / tf.cast(batch_size - 1, dtype="float32")

        num_features = tf.shape(ca)[0]

        off_diag_ca = self.off_diagonal(ca)
        off_diag_ca = tf.math.pow(off_diag_ca, 2)
        off_diag_ca = tf.math.reduce_sum(off_diag_ca) / num_features

        off_diag_cb = self.off_diagonal(cb)
        off_diag_cb = tf.math.pow(off_diag_cb, 2)
        off_diag_cb = tf.math.reduce_sum(off_diag_cb) / num_features

        # covariance loss(1d tensor) for redundancy reduction
        cov_loss = off_diag_ca + off_diag_cb

        loss = 25 * sim_loss + 25 * std_loss + cov_loss

        return loss

    def train(self, epochs, batch_size=8):
        return



