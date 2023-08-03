from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Model 
from tensorflow.keras.layers import Layer 
from tensorflow.keras.layers import *
from .factory import act_layer_factory, norm_layer_factory
from typing import *
import numpy as np


class MLP(tf.keras.layers.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

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
        super(MLP, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.drop_rate = drop_rate
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        act_layer = act_layer_factory(act_layer)

        self.fc1 = tf.keras.layers.Dense(
            units=hidden_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=mlp_bias,
            name="fc1",
        )
        self.act = act_layer()
        self.drop1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(
            units=projection_dim,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            use_bias=mlp_bias,
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
        config = super(MLP, self).get_config()
        config["hidden_dim"] = self.hidden_dim
        config["projection_dim"] = self.projection_dim
        config["drop_rate"] = self.drop_rate
        config["kernel_initializer"] = self.kernel_initializer
        config["bias_initializer"] = self.bias_initializer
        return config


class MlpHead(tf.keras.layers.Layer):
    """ MLP classification head
    """
    def __init__(self,
                 dim,
                 num_classes=1000,
                 mlp_ratio=4,
                 act_layer="square_relu",
                 norm_layer="layer_norm",
                 head_dropout=0.,
                 bias=True,
                 **kwargs
              ):

        super(MlpHead, self).__init__(**kwargs)
        self.dim = dim
        act_layer = act_layer_factory(act_layer)
        norm_layer = norm_layer_factory(norm_layer)
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = tf.keras.layers.Dense(hidden_features, use_bias=bias)
        self.act = act_layer()
        self.norm = norm_layer()
        self.fc2 = tf.keras.layers.Dense(num_classes, use_bias=bias)
        self.head_dropout = tf.keras.layers.Dropout(head_dropout)

    def call(self, x, training=False):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x, training=training)
        x = self.fc2(x)
        return x

    def get_config(self):
      config = super(MlpHead, self).get_config()
      config["dim"] = self.dim

      return config