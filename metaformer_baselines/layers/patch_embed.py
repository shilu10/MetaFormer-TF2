from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import Model 
from tensorflow.keras.layers import *
from .utils import get_initializer
from .factory import act_layer_factory, norm_layer_factory
from collections import * 
import collections
from ml_collections import ConfigDict
from typing import *
import numpy as np


class Stem(tf.keras.layers.Layer):
    """
    Stem implemented by a layer of convolution.
    Conv2d params constant across all models.
    """

    def __init__(
            self,
            out_channels,
            norm_layer=None,
            **kwargs
    ):
        super(Stem, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.norm = norm_layer_factory(norm_layer)(name="stem_norm") if norm_layer is not None else tf.identity
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=7,
            strides=4,
            padding="same"
        )
        #self.norm = norm_layer(name="stem_norm") if norm_layer else tf.identity

    def call(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

    def get_config(self):
        config = super(Stem, self).get_config()

        config["out_channels"] = self.out_channels

        return config