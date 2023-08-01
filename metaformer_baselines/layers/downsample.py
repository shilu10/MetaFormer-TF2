import tensorflow as tf 
import numpy as np 


class Downsampling(tf.keras.layers.Layer):
    """
    Downsampling implemented by a layer of convolution.
    """

    def __init__(
            self,
            hidden_dims,
            kernel_size,
            strides=1,
            padding="same",
            norm_layer=None,
    ):
        super(Downsampling, self).__init__()
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size
        self.norm = norm_layer_factory(norm_layer)(name="downsample_norm") if norm_layer is not None else tf.identity
        
        self.norm_layer = norm_layer
        self.conv = tf.keras.layers.Conv2D(
            filters=hidden_dims,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding
        )

    def call(self, x):
        
        x = self.norm(x)
        x = self.conv(x)
        return x

    def get_config(self):
        config = super(Downsampling, self).get_config()

        config["hidden_dims"] = self.hidden_dims
        config["kernel_size"] = self.kernel_size

        return config
