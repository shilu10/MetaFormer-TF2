from tensorflow import keras
import tensorflow as tf
import numpy as np


class RandomMixing(tf.keras.layers.Layer):
    def __init__(self, projection_dim=None, num_tokens=196,  **kwargs):
        super(RandomMixing, self).__init__(**kwargs)

        self.num_tokens = num_tokens

    def build(self, input_shape):
      self.random_matrix = self.add_weight(
          shape = (self.num_tokens, self.num_tokens)
      )

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, (B, H*W, C))
        x = tf.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = tf.reshape(x, (B, H, W, C))
        return x

    def get_config(self):
      config = super(RandomMixing, self).get_config()
      config['num_tokens'] = self.num_tokens

      return config