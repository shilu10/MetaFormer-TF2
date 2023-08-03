from tensorflow import keras
import tensorflow as tf
import numpy as np


class SepConv(tf.keras.layers.Layer):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
            self,
            projection_dim,
            num_tokens=None,
            expansion_ratio=2,
            act1_layer="star_relu",
            act2_layer=tf.identity,
            bias=False,
            kernel_size=7,
            padding=3,
            **kwargs
    ):
        super(SepConv, self).__init__()

        act1_layer = act_layer_factory(act1_layer)
        self.projection_dim = projection_dim

        mid_channels = int(expansion_ratio * projection_dim)
        self.pwconv1 = tf.keras.layers.Dense(
                                    units=mid_channels,
                                    use_bias=bias
                                  )
        self.act1 = act1_layer()
        self.dwconv = tf.keras.layers.Conv2D(filters=mid_channels,
                                             kernel_size=kernel_size,
                                             padding="same",
                                             groups=mid_channels,
                                             use_bias=bias)  # depthwise conv


        # self.dwconv =  tf.keras.layers.DepthwiseConv2D(
        #                       kernel_size=kernel_size,
        #                       padding="same"
        #                    )

        self.act2 = act2_layer
        self.pwconv2 = tf.keras.layers.Dense(
                                    units=projection_dim,
                                    use_bias=bias
                                  )

    def call(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config["projection_dim"] = projection_dim

        return config