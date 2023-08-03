from .utils import modify_tf_block
from .utils import *
from .metaformer_baselines.model import MetaFormer
from .metaformer_baselines.layers import *
from .metaformer_baselines.blocks import *
import numpy as np 
import os, sys, shutil
import tqdm 
import glob 
import pandas as pd 
import tensorflow as tf 
import tensorflow.keras as keras 
import argparse
import timm, transformers 
from typing import Dict, List
import yaml 
from imutils import paths
import torch
from .metaformer_baselines.layers import Pooling, SepConv, Attention, RandomMixing
from functools import partial


def port_weights(model_type="poolformerv2_s12", 
                model_savepath =".", 
                include_top=True,
                model_main_res_fname="main_result.csv"
              ):

    print("Intializing the Tensorflow Model")
    
    # read the data from yaml file
    config_file_path = f"configs/{model_type}.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    layer_scale_init_values = None
    res_scale_init_values = [None, None, 1.0, 1.0]

    tf_model = MetaFormer(
        depths = data.get('depths'),
        dims = data.get("dims"),
        norm_layers = data.get("block_norm"),
        output_norm = data.get('main_norm'),
        downsample_norm = data.get("downsample_norm"),
        layer_scale_init_values = layer_scale_init_values,
        res_scale_init_values = res_scale_init_values,
        include_top = include_top,
        mlp_bias = data.get("mlp_bias"),
        mlp_act = data.get("mlp_act"),
        use_mlp_head = data.get("use_mlp_head"),
        nchws = [True, True, True, True] if "caformer" not in model_type else [True, True, False, False],
        token_mixer = token_mixer_dict.get(data.get('token_mixer'))
    )

    dummy_input = np.zeros((1, 224, 224, 3))
    _ = tf_model(dummy_input)

    # path not exists 
    if not os.path.exists(model_main_res_fname):
      make_model_res_file(model_main_res_fname)

    # calculating the flops and nb_params
    #nb_flops = tf_model.flops()
    #nb_flops = int(convert_kb_to_gb(nb_flops))

    nb_params = tf_model.count_params()
    nb_params = int(nb_params / 1000000)

    add_model_res(model_main_res_fname, model_type, nb_params)

    print('Loading the Pytorch model!!!')

    # pt_model_dict
    pt_model_dict = get_pt_model_dict(model_type)

    # port weights
    if "poolformerv2" in model_type:
      modify_poolformerv2(tf_model, pt_model_dict, include_top)

    elif "randformer" in model_type:
      modify_identity_rand_former(tf_model, pt_model_dict, include_top)

    elif "identityformer" in model_type:
      modify_identity_rand_former(tf_model, pt_model_dict, include_top)

    else:
      modify_conv_ca_former(tf_model, pt_model_dict, include_top)

    
    save_path = os.path.join(model_savepath, model_type)
    save_path = f"{save_path}_fe" if not include_top else save_path
    tf_model.save(save_path)
    
    print(f"TensorFlow model serialized at: {save_path}...")


def modify_poolformerv2_stage(stage, stage_indx, pt_model_dict):
  
  block_indx = 0
  for block in stage.layers:
    pt_block_name = f"stages.{stage_indx}.blocks.{block_indx}"

    if isinstance(block, MetaFormerBlock):
      # normalization
      block.norm1 = modify_tf_block(
          tf_component = block.norm1,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm1.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm1.bias"]
      )

      block.norm2 = modify_tf_block(
          tf_component = block.norm2,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm2.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
      )

      # mlp layer
      block.mlp.fc1 = modify_tf_block(
          tf_component = block.mlp.fc1,
          pt_weight = pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"]
      )

      block.mlp.fc2 = modify_tf_block(
          tf_component = block.mlp.fc2,
          pt_weight = pt_model_dict[f"{pt_block_name}.mlp.fc2.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc2.bias"]
      )

      # mlp act scale and bias
      block.mlp.act.scale.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.scale"]))
      block.mlp.act.bias.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.bias"]))

      # res_scale and layer_scale
      if (block.res_scale_1) is not None:
        block.res_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale1.scale"]))

      if (block.res_scale_2) is not None:
        block.res_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale2.scale"]))

      if (block.layer_scale_1) is not None:

        block.layer_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale1.scale"]))

      if (block.layer_scale_2) is not None:
        block.layer_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale2.scale"]))

      block_indx += 1

    if isinstance(block, Downsampling):
      block.conv = modify_tf_block(
          tf_component = block.conv,
          pt_weight = pt_model_dict[f"stages.{stage_indx}.downsample.conv.weight"],
          pt_bias = pt_model_dict[f"stages.{stage_indx}.downsample.conv.bias"]
      )

      block.norm = modify_tf_block(
          tf_component = block.norm,
          pt_weight = pt_model_dict[f"stages.{stage_indx}.downsample.norm.weight"],
         # pt_bias = pt_model_dict[f"stages.{stage_indx}.downsample.norm.bias"]
      )


def modify_identity_rand_former_stage(stage, stage_indx, pt_model_dict):
  block_indx = 0
  for block in stage.layers:
    pt_block_name = f"stages.{stage_indx}.{block_indx}"

    if isinstance(block, MetaFormerBlock):
      # normalization
      block.norm1 = modify_tf_block(
          tf_component = block.norm1,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm1.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm1.bias"]
      )

      block.norm2 = modify_tf_block(
          tf_component = block.norm2,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm2.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
      )

      # mlp layer
      arr = np.expand_dims(pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"], axis=-1)
      arr = np.expand_dims(arr, axis=-1)
      block.mlp.fc1 = modify_tf_block(
          tf_component = block.mlp.fc1,
          pt_weight = arr,
          #pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"]
      )

      arr = np.expand_dims(pt_model_dict[f"{pt_block_name}.mlp.fc2.weight"], axis=-1)
      arr = np.expand_dims(arr, axis=-1)
      block.mlp.fc2 = modify_tf_block(
          tf_component = block.mlp.fc2,
          pt_weight = arr,
          #pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc2.bias"]
      )

      # mlp act scale and bias
      block.mlp.act.scale.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.scale"]))
      block.mlp.act.bias.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.bias"]))

      # res_scale and layer_scale
      
      if (block.res_scale_1) is not None:
        block.res_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale1.scale"]))

      if (block.res_scale_2) is not None:
        block.res_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale2.scale"]))

      if (block.layer_scale_1) is not None:
        block.layer_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale1.scale"]))

      if (block.layer_scale_2) is not None:
        block.layer_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale2.scale"]))

      # random mixing for randformer
      if (block.token_mixer != tf.identity) and isinstance(block.token_mixer, RandomMixing):
        block.token_mixer.random_matrix.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.token_mixer.random_matrix"]))

      block_indx += 1

    if isinstance(block, Downsampling):
      block.conv = modify_tf_block(
          tf_component = block.conv,
          pt_weight = pt_model_dict[f"downsample_layers.{stage_indx}.conv.weight"],
          pt_bias = pt_model_dict[f"downsample_layers.{stage_indx}.conv.bias"]
      )

      block.norm = modify_tf_block(
          tf_component = block.norm,
          pt_weight = pt_model_dict[f"downsample_layers.{stage_indx}.pre_norm.weight"],
         # pt_bias = pt_model_dict[f"stages.{stage_indx}.downsample.norm.bias"]
      )


def modify_poolformerv2(tf_model, pt_model_dict, include_top):

  # patch embed (stem) conv and norm
  tf_model.layers[0].conv = modify_tf_block(
            tf_component = tf_model.layers[0].conv,
            pt_weight = pt_model_dict["stem.conv.weight"],
            pt_bias = pt_model_dict["stem.conv.bias"]
        )

  tf_model.layers[0].norm = modify_tf_block(
            tf_component = tf_model.layers[0].norm,
            pt_weight = pt_model_dict["stem.norm.weight"],
            #pt_bias = pt_model_dict["stem.conv.bias"]
        )

  # main norm
  tf_model.layers[-3] = modify_tf_block(
            tf_component = tf_model.layers[-3],
            pt_weight = pt_model_dict["head.norm.weight"],
            pt_bias = pt_model_dict["head.norm.bias"]
        )

  # head
  if include_top:
    tf_model.layers[-1] = modify_tf_block(
              tf_component = tf_model.layers[-1],
              pt_weight = pt_model_dict["head.fc.weight"],
              pt_bias = pt_model_dict["head.fc.bias"]
          )
  
  # modify poolformerv2 stages
  for idx, stage in enumerate(tf_model.layers[1: 1+4]):
    modify_poolformerv2_stage(stage, idx, pt_model_dict)


def modify_identity_rand_former(tf_model, pt_model_dict, include_top):
  # patch embed (stem) conv and norm

  tf_model.layers[0].conv = modify_tf_block(
            tf_component = tf_model.layers[0].conv,
            pt_weight = pt_model_dict["downsample_layers.0.conv.weight"],
            pt_bias = pt_model_dict["downsample_layers.0.conv.bias"]
        )

  tf_model.layers[0].norm = modify_tf_block(
            tf_component = tf_model.layers[0].norm,
            pt_weight = pt_model_dict["downsample_layers.0.post_norm.weight"],
            #pt_bias = pt_model_dict["stem.conv.bias"]
        )

  # main norm
  tf_model.layers[-3] = modify_tf_block(
            tf_component = tf_model.layers[-3],
            pt_weight = pt_model_dict["norm.weight"],
            pt_bias = pt_model_dict["norm.bias"]
        )
  
  # head
  if include_top:
    tf_model.layers[-1] = modify_tf_block(
              tf_component = tf_model.layers[-1],
              pt_weight = pt_model_dict["head.weight"],
              pt_bias = pt_model_dict["head.bias"]
          )
  
  # modify identity and rand former stages
  for idx, stage in enumerate(tf_model.layers[1: 1+4]):
    modify_identity_rand_former_stage(stage, idx, pt_model_dict)


def modify_conv_ca__stage(stage, stage_indx, pt_model_dict):

  block_indx = 0
  for block in stage.layers:
    pt_block_name = f"stages.{stage_indx}.blocks.{block_indx}"
    if isinstance(block, MetaFormerBlock):
      # normalization
      block.norm1 = modify_tf_block(
          tf_component = block.norm1,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm1.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm1.bias"]
      )

      block.norm2 = modify_tf_block(
          tf_component = block.norm2,
          pt_weight = pt_model_dict[f"{pt_block_name}.norm2.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
      )

      # mlp layer
      block.mlp.fc1 = modify_tf_block(
          tf_component = block.mlp.fc1,
          pt_weight = pt_model_dict[f"{pt_block_name}.mlp.fc1.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc1.bias"]
      )

      block.mlp.fc2 = modify_tf_block(
          tf_component = block.mlp.fc2,
          pt_weight = pt_model_dict[f"{pt_block_name}.mlp.fc2.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.mlp.fc2.bias"]
      )

      # mlp act scale and bias
      block.mlp.act.scale.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.scale"]))
      block.mlp.act.bias.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.mlp.act.bias"]))

      # res_scale and layer_scale
      if (block.res_scale_1) is not None:
        block.res_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale1.scale"]))

      if (block.res_scale_2) is not None:
        block.res_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.res_scale2.scale"]))

      if (block.layer_scale_1) is not None:

        block.layer_scale_1.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale1.scale"]))

      if (block.layer_scale_2) is not None:
        block.layer_scale_2.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.layer_scale2.scale"]))

      # for convformer using sepconv mixer
      if isinstance(block.token_mixer, SepConv):
        # act1 -> starrelu - bias and scale, pwconv1, pwconv2, dwconv
        # for token_mixer act1
        block.token_mixer.act1.scale.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.token_mixer.act1.scale"]))
        block.token_mixer.act1.bias.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.token_mixer.act1.bias"]))

        # for token_mixer pwconv1 and pwconv2
        block.token_mixer.pwconv1 = modify_tf_block(
          tf_component = block.token_mixer.pwconv1,
          pt_weight = pt_model_dict[f"{pt_block_name}.token_mixer.pwconv1.weight"][:, :, 0, 0],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
        )

        block.token_mixer.pwconv2 = modify_tf_block(
          tf_component = block.token_mixer.pwconv2,
          pt_weight = pt_model_dict[f"{pt_block_name}.token_mixer.pwconv2.weight"][:, :, 0, 0],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
        )

        # token_mixer dwconv
        block.token_mixer.dwconv = modify_tf_block(
          tf_component = block.token_mixer.dwconv,
          pt_weight = pt_model_dict[f"{pt_block_name}.token_mixer.dwconv.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
        )

      if isinstance(block.token_mixer, Attention):
        # qkv vector
        block.token_mixer.qkv = modify_tf_block(
            block.token_mixer.qkv,
            pt_model_dict[f"{pt_block_name}.token_mixer.qkv.weight"]
        )

        # attn projection
        block.token_mixer.proj = modify_tf_block(
            block.token_mixer.proj,
            pt_model_dict[f"{pt_block_name}.token_mixer.proj.weight"]
        )

      block_indx += 1

    if isinstance(block, Downsampling):
      block.conv = modify_tf_block(
          tf_component = block.conv,
          pt_weight = pt_model_dict[f"stages.{stage_indx}.downsample.conv.weight"],
          pt_bias = pt_model_dict[f"stages.{stage_indx}.downsample.conv.bias"]
      )

      block.norm = modify_tf_block(
          tf_component = block.norm,
          pt_weight = pt_model_dict[f"stages.{stage_indx}.downsample.norm.weight"],
         # pt_bias = pt_model_dict[f"stages.{stage_indx}.downsample.norm.bias"]
      )


def modify_conv_ca_former(tf_model, pt_model_dict, include_top):

  # stem layer
  tf_model.layers[0].conv = modify_tf_block(
          tf_component = tf_model.layers[0].conv,
          pt_weight = pt_model_dict["stem.conv.weight"],
          pt_bias = pt_model_dict["stem.conv.bias"]
      )

  tf_model.layers[0].norm = modify_tf_block(
            tf_component = tf_model.layers[0].norm,
            pt_weight = pt_model_dict["stem.norm.weight"],
            #pt_bias = pt_model_dict["stem.conv.bias"]
        )

  # main norm
  tf_model.layers[-3] = modify_tf_block(
            tf_component = tf_model.layers[-3],
            pt_weight = pt_model_dict["head.norm.weight"],
            pt_bias = pt_model_dict["head.norm.bias"]
        )

  # headfc1
  if include_top:
    tf_model.layers[-1].fc1 = modify_tf_block(
              tf_component = tf_model.layers[-1].fc1,
              pt_weight = pt_model_dict["head.fc.fc1.weight"],
              pt_bias = pt_model_dict["head.fc.fc1.bias"]
          )

    tf_model.layers[-1].fc2 = modify_tf_block(
              tf_component = tf_model.layers[-1].fc2,
              pt_weight = pt_model_dict["head.fc.fc2.weight"],
              pt_bias = pt_model_dict["head.fc.fc2.bias"]
          )

    # head-norm
    tf_model.layers[-1].norm = modify_tf_block(
              tf_component = tf_model.layers[-1].norm,
              pt_weight = pt_model_dict["head.fc.norm.weight"],
              pt_bias = pt_model_dict["head.fc.norm.bias"]
          )
  
  # modify conv and ca stages
  for idx, stage in enumerate(tf_model.layers[1: 1+4]):
    modify_conv_ca__stage(stage, idx, pt_model_dict)


def get_pt_model_dict(model_type):
  if "rand" in model_type or "identity" in model_type:
    weight_dict = torch.hub.load_state_dict_from_url(url=urls["model_type"], 
                                                    map_location="cpu", 
                                                    check_hash=True
                                                  )
    pt_model_dict = {k: np.array(weight_dict[k]) for k in weight_dict.keys()}

  else: 
    pt_model = timm.create_model(
      model_name = model_type,
      pretrained = True,
      num_classes = 1000
    )
    pt_model.eval()
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: np.array(pt_model_dict[k]) for k in pt_model_dict.keys()}
  
  return pt_model_dict


def make_model_res_file(fpath):
  with open(fpath, "w") as file:
    file.write("model_variant, #params\n")


def add_model_res(fpath, model_variant, params):
  with open(fpath, "a") as file:
    file.write(f"{model_variant}, {params}M\n")


def convert_kb_to_gb(val):
  gb_val = val / 1000 / 1000 / 1000
  return gb_val  


urls = {
  'randformer_s12': 'https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s12.pth',
  'randformer_s24': 'https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s24.pth',
  'randformer_s36': 'https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s36.pth',
  'randformer_m36': 'https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m36.pth',
  'randformer_m48': 'https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m48.pth',

  'identityformer_s12': 'https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s12.pth',
  'identityformer_s24': 'https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s24.pth',
  'identityformer_s36': 'https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s36.pth',
  'identityformer_m36': 'https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m36.pth',
  'identityformer_m48': 'https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m48.pth',
}   


token_mixer_dict = {
  "pooling": Pooling,
  "identity": tf.identity,
  "random_mixer": [tf.identity, tf.identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
  "sepconv": SepConv,
  "ca": [SepConv, SepConv, Attention, Attention],
}