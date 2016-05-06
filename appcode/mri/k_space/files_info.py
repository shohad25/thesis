"""
All k_space files info
"""
import numpy as np
import os
import json

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'meta_data_params.json'), 'r') as f:
    meta_data_params = json.load(f)
    sizeof_meta_data = meta_data_params["sizeof_meta_data"]

files_info = {"k_space_real":  {"width": 256, "height": 128, "channels": 1, "dtype": np.float32},
              "k_space_imag":  {"width": 256, "height": 128, "channels": 1, "dtype": np.float32},
              "image":         {"width": 256, "height": 128, "channels": 1, "dtype": np.int16},

              "k_space_real_gt":  {"width": 256, "height": 256, "channels": 1, "dtype": np.float32},
              "k_space_imag_gt":  {"width": 256, "height": 256, "channels": 1, "dtype": np.float32},
              "image_gt":         {"width": 256, "height": 256, "channels": 1, "dtype": np.int16},

              "mask":          {"width": 256, "height": 256, "channels": 1, "dtype": np.uint8},
              "meta_data":     {"width": sizeof_meta_data, "height": 1,   "channels": 1, "dtype": np.float32}}


def get_file_info(file_name):
    return files_info[file_name]
