from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LayerNormalization, GroupNormalization


def act_layer_factory(act_layer: str):
    """Returns a function that creates the required activation layer."""
    if act_layer in {"linear", "swish", "relu", "gelu", "sigmoid"}:
        return lambda **kwargs: tf.keras.layers.Activation(act_layer, **kwargs)
    if act_layer == "relu6":
        return lambda **kwargs: tf.keras.layers.ReLU(max_value=6, **kwargs)

    if act_layer == "star_relu":
      return StarReLU

    if act_layer == "square_relu":
      return SquaredReLU

    else:
        raise ValueError(f"Unknown activation: {act_layer}.")


def norm_layer_factory(norm_layer: str):
    """Returns a function that creates a normalization layer"""
    if norm_layer == "":
        return lambda **kwargs: tf.keras.layers.Activation("linear", **kwargs)

    elif norm_layer == "batch_norm":
        bn_class = tf.keras.layers.BatchNormalization
        bn_args = {
            "momentum": 0.9,  # We use PyTorch default args here
            "epsilon": 1e-5,
        }
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "batch_norm_tf":  # Batch norm with TF default for epsilon
        bn_class = tf.keras.layers.BatchNormalization
        bn_args = {
            "momentum": 0.9,
            "epsilon": 1e-3,
        }
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "layer_norm":
        bn_class = tf.keras.layers.LayerNormalization
        bn_args = {"epsilon": 1e-5}  # We use PyTorch default args here
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "layer_norm_eps_1e-6":
        bn_class = tf.keras.layers.LayerNormalization
        bn_args = {"epsilon": 1e-6}
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "affine":
        return Affine

    elif norm_layer == "group_norm":
        return GroupNormalization

    elif norm_layer == "group_norm_1grp":
        # Group normalization with one group. Used by PoolFormer.
        bn_class = GroupNormalization
        bn_args = {"groups": 1}
        return lambda **kwargs: bn_class(**bn_args, **kwargs)

    elif norm_layer == "layer_norm_nobias":
      bn_class = LayerNormalizationNoBias
      bn_args = {"epsilon": 1e-5}  # We use PyTorch default args here
      return lambda **kwargs: bn_class(**bn_args, **kwargs)

    else:
        raise ValueError(f"Unknown normalization layer: {norm_layer}")


class LayerNormalizationNoBias(keras.layers.Layer):
    def __init__(self, epsilon=0.001, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, batch_input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=batch_input_shape[-1:],
            dtype=tf.float32,
            initializer="ones")

        super().build(batch_input_shape)

    def call(self, X):
        mean, variance = tf.nn.moments(X, axes=[-1], keepdims=True)
        return self.alpha * (X - mean) / (tf.sqrt(variance + self.epsilon))

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "eps": self.epsilon}


class SquaredReLU(tf.keras.layers.Layer):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """

    def __init__(self, inplace=False):
        super(SquaredReLU, self).__init__()
        self.relu = tf.keras.layers.ReLU()

    def forward(self, x):
        return tf.math.square(self.relu(x))


class StarReLU(tf.keras.layers.Layer):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):

        super().__init__()
        self.inplace = inplace
        self.relu = tf.keras.layers.ReLU()
        self.scale = tf.Variable(scale_value * tf.ones(1),
            trainable=scale_learnable)
        self.bias = tf.Variable(bias_value * tf.ones(1),
            trainable=bias_learnable)

    def call(self, x):
        return self.scale * self.relu(x)**2 + self.bias