"""
All k_space files info
"""
import numpy as np

files_info = {"k_space_real":  {"width": 256, "height": 256, "channels": 1, "dtype": np.int16},
              "k_space_imag":  {"width": 256, "height": 256, "channels": 1, "dtype": np.int16},
              "image":         {"width": 256, "height": 256, "channels": 1, "dtype": np.int16},
              "mask":          {"width": 256, "height": 256, "channels": 1, "dtype": np.uint8},
              "meta_data":     {"width": 1, "height": 1,   "channels": 1, "dtype": np.float32}}


def get_file_info(file_name):
    return files_info[file_name]
