"""
MRI data base - all needed for handle my MRI database, NIfTI and DICOM as well
"""
import os
import nibabel as nib
import numpy as np
import json

with open('data_to_path.json', 'r') as f:
    data_to_path = json.load(f)


class MriDataBase:
    """
    Mri data base class - only source data, for nifty or dicom
    """

    def __init__(self, data_name, nifty_or_dicom='nifty'):
        """
        Constructor
        :param data_name: data name
        :param nifty_or_dicom: nifty / dicom
        :return: MriDataBase object
        """
        self.data_name = data_name
        self.nifty_or_dicom = nifty_or_dicom
        self.data_path = data_to_path[data_name]["data"]
        self.labels_path = data_to_path[data_name]["labels"]
        file_suffix = "hdr" if nifty_or_dicom == 'nifty' else 'dicom'  # TODO
        self.items = [item for item in os.listdir(data_to_path[data_name]["data"]) if file_suffix in item]