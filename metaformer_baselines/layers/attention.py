import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
from typing import * 


class Attention(tf.keras.layers.Layer):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, projection_dim, num_tokens=None, 
                                    head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else projection_dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = tf.keras.layers.Dense(self.attention_dim * 3,
                                         use_bias=qkv_bias,
                                         name="qkv_vector"
                                      )

        self.attn_drop = tf.keras.layers.Dropout(attn_drop)
        self.proj = tf.keras.layers.Dense(projection_dim,
                                          use_bias=proj_bias,
                                          name="proj_vector"
                                        )
        self.proj_drop = tf.keras.layers.Dropout(proj_drop)


    def call(self, x):
        B, N, C = x.shape
        x = tf.cast(x, dtype=tf.float32)
        qkv = tf.transpose(tf.reshape(self.qkv(x), (-1, N, 3, self.num_heads, self.head_dim)), perm=(2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv)  

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)

        x = tf.reshape(tf.transpose(x, (0, 2, 1, 3)), (-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_config(self):
      config = super(Attention, self).get_config()
      config["head_dim"] = self.head_dim
      config['scale'] = self.scale

      return config