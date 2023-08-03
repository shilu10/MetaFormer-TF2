import tensorflow as tf 
import numpy as np 
from ..layers import DropPath, ConvMLP, MLP
from ..layers import norm_layer_factory, act_layer_factory



class MetaFormerBlock(tf.keras.layers.Layer):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(
            self,
            projection_dims,
            token_mixer=Pooling,
            mlp_act="gelu",
            mlp_bias=False,
            norm_layer="layer_norm",
            proj_drop=0.,
            drop_path=0.,
            use_nchw=True,
            layer_scale_init_value=None,
            res_scale_init_value=None,
            **kwargs
    ):
        super(MetaFormerBlock, self).__init__(**kwargs)
        self.projection_dims = projection_dims
        self.layer_scale_init_value = layer_scale_init_value
        self.res_scale_init_value = res_scale_init_value
        norm_layer = norm_layer_factory(norm_layer)

        self.norm1 = norm_layer(name="norm1")
        self.token_mixer = token_mixer(projection_dim=projection_dims) if not token_mixer == tf.identity else token_mixer
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.layer_scale_1 = None
        self.res_scale_1 = None

        self.norm2 = norm_layer(name="norm2")

        if use_nchw:
          self.mlp = ConvMLP(
              int(4 * projection_dims),
              projection_dims,
              act_layer=mlp_act,
              drop_rate=proj_drop,
              mlp_bias = mlp_bias
          )
        else:
          self.mlp = MLP(
              int(4 * projection_dims),
              projection_dims,
              act_layer=mlp_act,
              drop_rate=proj_drop,
              mlp_bias = mlp_bias
          )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else tf.identity
        self.layer_scale_2 = None
        self.res_scale_2 = None

    def call(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.token_mixer(x)
        x = self.drop_path1(x)

        if self.layer_scale_1 is not None:
          x = self.layer_scale_1 * x

        if self.res_scale_1 is not None:
          shortcut = (self.res_scale_1 * shortcut)

        x = shortcut + x
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path1(x)

        if self.layer_scale_2 is not None:
          x = self.layer_scale_2 * x

        if self.res_scale_2 is not None:
          shortcut = (self.res_scale_2 * shortcut)

        x = shortcut + x

        return x

    def build(self, input_shape):

        if self.layer_scale_init_value is not None:
          self.layer_scale_1 = self.add_weight(
              shape=(self.projection_dims,),
              initializer=tf.keras.initializers.Constant(value=self.layer_scale_init_value),
              trainable=True,
              name="layer_scale_1",
          )
          self.layer_scale_2 = self.add_weight(
              shape=(self.projection_dims,),
              initializer=tf.keras.initializers.Constant(value=self.layer_scale_init_value),
              trainable=True,
              name="layer_scale_2",
          )

        if self.res_scale_init_value is not None:
          self.res_scale_1 = self.add_weight(
              shape=(self.projection_dims,),
              initializer=tf.keras.initializers.Constant(value=self.res_scale_init_value),
              trainable=True,
              name="layer_scale_1",
          )
          self.res_scale_2 = self.add_weight(
              shape=(self.projection_dims,),
              initializer=tf.keras.initializers.Constant(value=self.res_scale_init_value),
              trainable=True,
              name="layer_scale_2",
          )

    def get_config(self):
      config = super(MetaFormerBlock, self).get_config()

      config['projection_dims'] = self.projection_dims
      config['layer_scale_init_value'] = self.layer_scale_init_value
      config['res_scale_init_value'] = self.res_scale_init_value

      return config
