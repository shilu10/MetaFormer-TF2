# metaformer baselines

def modify_metaformer_stage(stage, stage_indx, pt_model_dict):
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

      # for convformer using sepconv mixer
      if isinstance(block.token_mixer, SepConv):
        # act1 -> starrelu - bias and scale, pwconv1, pwconv2, dwconv
        # for token_mixer act1
        block.token_mixer.act1.scale.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.token_mixer.act1.scale"]))
        block.token_mixer.act1.bias.assign(tf.Variable(pt_model_dict[f"{pt_block_name}.token_mixer.act1.bias"]))

        # for token_mixer pwconv1 and pwconv2
        block.token_mixer.pwconv1 = modify_tf_block(
          tf_component = block.token_mixer.pwconv1,
          pt_weight = pt_model_dict[f"{pt_block_name}.token_mixer.pwconv1.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
      )
        
        block.token_mixer.pwconv2 = modify_tf_block(
          tf_component = block.token_mixer.pwconv2,
          pt_weight = pt_model_dict[f"{pt_block_name}.token_mixer.pwconv2.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
      )
        
        # token_mixer dwconv
        block.token_mixer.dwconv = modify_tf_block(
          tf_component = block.token_mixer.dwconv,
          pt_weight = pt_model_dict[f"{pt_block_name}.token_mixer.dwconv.weight"],
          #pt_bias = pt_model_dict[f"{pt_block_name}.norm2.bias"]
      )

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



3## head layers 

# for sail-sg weight conversion and use_mlp_head = true
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

# headfc1
tf_model.layers[-1].fc1 = modify_tf_block(
          tf_component = tf_model.layers[-1].fc1,
          pt_weight = pt_model_dict["head.fc1.weight"],
          pt_bias = pt_model_dict["head.fc1.bias"]
      )

tf_model.layers[-1].fc2 = modify_tf_block(
          tf_component = tf_model.layers[-1].fc2,
          pt_weight = pt_model_dict["head.fc2.weight"],
          pt_bias = pt_model_dict["head.fc2.bias"]
      )

# head-norm
tf_model.layers[-1].norm = modify_tf_block(
          tf_component = tf_model.layers[-1].norm,
          pt_weight = pt_model_dict["head.norm.weight"],
          pt_bias = pt_model_dict["head.norm.bias"]
      )
