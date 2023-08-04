import os 
import shutil 
import glob 
import numpy as np 
import yaml 
from imutils import paths 
from .convert import port_weights

# all config files 
all_model_types = [
    'poolformerv2_s12',
    'poolformerv2_s24',
    'poolformerv2_s36',
    'poolformerv2_m36',
    'poolformerv2_m48',
    'identityformer_s12',
    'identityformer_s24',
    'identityformer_s36',
    'identityformer_m36',
    'identityformer_m48',
    'randformer_s12',
    'randformer_s24',
    'randformer_s36',
    'randformer_m36',
    'randformer_m48',
    'convformer_s18',
    'convformer_s36',
    'convformer_m36',
    'convformer_b36',
    'caformer_s18',
    'caformer_s36',
    'caformer_m36',
    'caformer_b36'
]

def main(model_savepath="models/"):

    try:
        config_file_paths = list(paths.list_files("configs/"))
        for config_file_path in all_model_types:
            # porting all model types from pytorch to tensorflow
            try:
                model_type = config_file_path.split("/")[-1].split(".")[0]
                print(f"Processing the  model type: {model_type}\n")

                port_weights(
                    model_type=model_type,
                    model_savepath=model_savepath,
                    include_top=True
                )    
            
            except Exception as err:
                print("This specific model_type cannot be ported", err)

    except Exception as err:
        return err
