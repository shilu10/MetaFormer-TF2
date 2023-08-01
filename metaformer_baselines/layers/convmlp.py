import tensorflow as tf 
import numpy as np 
from collections import * 
import collections 
from .factory import act_layer_factory

class ConvMLP(tf.keras.layers.Layer):
    """
    ConvMLP is the same block as MLP, but it uses 1x1 convolutions instead of fully
    connected layers.

    Used in ``ConvNeXt`` models.
    """

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        drop_rate: float,
        act_layer: str,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        mlp_bias: bool = False,
        **kwargs,
    ):
        super(ConvMLP, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Conv2D(
            filters=hidden_dim,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            use_bias = mlp_bias,
            bias_initializer=bias_initializer,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Conv2D(
            filters=projection_dim,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            use_bias = mlp_bias,
            bias_initializer=bias_initializer,
            name="fc2",
        )
        self.drop2 = tf.keras.layers.Dropout(rate=drop_rate)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)
        return x

    def get_config(self):
        config = super(ConvMLP, self).get_config()
        
        config["hidden_dim"] = self.hidden_dim
        config["projection_dim"] = self.projection_dim

        return config
