import tensorflow as tf 
import numpy as np 
from ..layers import Downsampling
from .poolformer_block import PoolFormerBlock

class PoolFormerStage(tf.keras.Model):

    def __init__(
            self,
            in_chs,
            out_chs,
            depth=2,
            token_mixer=tf.identity,
            mlp_act="relu",
            mlp_bias=False,
            downsample_norm="layer_norm",
            norm_layer="layer_norm",
            proj_drop=0.,
            dp_rates=[0.] * 2,
            layer_scale_init_value=None,
            res_scale_init_value=None,
            **kwargs,
    ):
        super(PoolFormerStage, self).__init__(**kwargs)

        #self.grad_checkpointing = False
        self.use_nchw = True
        self.in_chs = in_chs
        self.out_chs = out_chs

        # don't downsample if in_chs and out_chs are the same
        self.downsample = tf.identity if in_chs == out_chs else Downsampling(
            out_chs,
            kernel_size=3,
            strides=2,
            padding="same",
            norm_layer=downsample_norm,
        )

        self.blocks = ([PoolFormerBlock(
            projection_dims=out_chs,
            token_mixer=token_mixer,
            mlp_act=mlp_act,
            mlp_bias=mlp_bias,
            norm_layer=norm_layer,
            proj_drop=proj_drop,
            drop_path=dp_rates[i],
            layer_scale_init_value=layer_scale_init_value,
            res_scale_init_value=res_scale_init_value,
            use_nchw=self.use_nchw,
            name = f"block_{i}"
        ) for i in range(depth)])


    def call(self, x):
        x = self.downsample(x)
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        if not self.use_nchw:
            x = x.reshape(B, C, -1).transpose(1, 2)

        else:
          for block in self.blocks:
            x = block(x)

        if not self.use_nchw:
            x = x.transpose(1, 2).reshape(B, C, H, W)

        return x

    def get_config(self):
        config = super(PoolFormerStage, self).get_config()

        config["in_chs"] = self.in_chs
        config['out_chs'] = self.out_chs
        config["use_nchw"] = self.use_nchw

        return config
