import tensorflow as tf 
import numpy as np 
from .blocks import MetaFormerStage 
from .layers import norm_layer_factory, act_layer_factory, Stem
from .layers import Pooling, MlpHead, MLP


class MetaFormer(tf.keras.Model):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels.
        num_classes (int): Number of classes for classification head.
        global_pool: Pooling for classifier head.
        depths (list or tuple): Number of blocks at each stage.
        dims (list or tuple): Feature dimension at each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage.
        mlp_act: Activation layer for MLP.
        mlp_bias (boolean): Enable or disable mlp bias term.
        drop_path_rate (float): Stochastic depth rate.
        drop_rate (float): Dropout rate.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for res Scale on residual connections.
            None means not use the res scale. From: https://arxiv.org/abs/2110.09456.
        downsample_norm (nn.Module): Norm layer used in stem and downsampling layers.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage.
        output_norm: Norm layer before classifier head.
        use_mlp_head: Use MLP classification head.
    """

    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            depths=(2, 2, 6, 2),
            dims=(64, 128, 320, 512),
            token_mixers=Pooling,
            mlp_act="gelu",
            mlp_bias=True,
            drop_path_rate=0.,
            proj_drop_rate=0.,
            drop_rate=0.0,
            layer_scale_init_values=1e-5,
            res_scale_init_values=None,
            downsample_norm=None,
            norm_layers="group_norm_1grp",
            output_norm="layer_norm",
            include_top=True,
            use_mlp_head=False,
            nchws=[True, True, True, True],
            **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = dims[-1]
        self.drop_rate = drop_rate
        self.num_stages = len(depths)
        self.include_top = include_top
        output_norm = norm_layer_factory(output_norm)

        # convert everything to lists if they aren't indexable
        if not isinstance(depths, (list, tuple)):
            depths = [depths]  # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * self.num_stages
        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * self.num_stages
        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * self.num_stages
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * self.num_stages


        print(token_mixers)

        # stages and stem layer
        self.feature_info = []
        self.stem = Stem(
            dims[0],
            norm_layer=downsample_norm,
            name="stem"
        )

        stages = []
        prev_dim = dims[0]
        dpr = np.linspace(0.0, drop_path_rate, sum(depths))
        dp_rates = np.split(dpr, np.cumsum(depths))

        for i in range(self.num_stages):
            print(token_mixers[i])
            stages += [MetaFormerStage(
                prev_dim,
                dims[i],
                depth=depths[i],
                token_mixer=token_mixers[i],
                mlp_act=mlp_act,
                mlp_bias=mlp_bias,
                proj_drop=proj_drop_rate,
                dp_rates=dp_rates[i],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
                downsample_norm=downsample_norm,
                norm_layer=norm_layers[i],
                use_nchw=nchws[i],
                name = f"stage_{i}",
                **kwargs,
            )]
            prev_dim = dims[i]
            self.feature_info += [dict(num_chs=dims[i], reduction=2, module=f'stages.{i}')]

        self.stages = stages

        # remaining layers
        self.norm = output_norm(name="main_norm")
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        if self.include_top and not use_mlp_head:
          self.head = (
              tf.keras.layers.Dense(units=num_classes, name="classification_head")
              if num_classes > 0
              else tf.keras.layers.Activation("linear")  # Identity layer
          )

        if self.include_top and use_mlp_head:
          self.head = MlpHead(dim=dims[-1], num_classes=num_classes, name="classification_head")

    def forward_head(self, x, include_top: bool = True):
        x = self.norm(x)
        x = self.pool(x)
        return x if not include_top else self.head(x)

    def forward_features(self, x):
        x = self.stem(x)

        for stage in self.stages:
          x = stage(x)

        return x

    def call(self, x):

        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def get_config(self):
        config = super(MetaFormer, self).get_config()

        config["num_classes"] = self.num_classes
        config["drop_rate"] = self.drop_rate
        config["include_top"] = self.include_top

        return config
