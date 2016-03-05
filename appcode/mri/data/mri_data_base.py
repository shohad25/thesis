"""
MRI data base - all needed for handle my MRI database, NIfTI and DICOM as well
"""
import json
import os
from collections import OrderedDict

import nibabel as nib

from common.viewers.orthoslicer import OrthoSlicer3D

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_to_path.json'), 'r') as f:
    data_to_path = json.load(f)


class MriDataBase:
    """
    Mri data base class - only source data, for nifty or dicom
    """

    def __init__(self, data_name, nifti_or_dicom='nifti'):
        """
        Constructor
        :param data_name: data name
        :param nifti_or_dicom: nifty / dicom
        :return: MriDataBase object
        """
        self.data_name = data_name
        self.nifty_or_dicom = nifti_or_dicom
        self.data_path = data_to_path[data_name]["data"]
        self.labels_path = data_to_path[data_name]["labels"]
        file_suffix = "hdr" if nifti_or_dicom == 'nifti' else 'dicom'  # TODO
        self.items = sorted([item for item in os.listdir(data_to_path[data_name]["data"]) if file_suffix in item])
        self.data = dict(img=[],meta_data=[], labels=[])
        self.info = self.set_info(data_to_path[data_name]["info"])

    def get_source_data(self, item='all'):
        """
        Read item from data path. If item=='all', read all items
        :param item: item name or all
        :return: dictionary
        """

        items = self.items if item=='all' else [item]
        data = dict(img=[], meta_data=[], labels=[])

        for it in items:
            if self.nifty_or_dicom == 'nifti':
                niftii_obj = nib.load(os.path.join(self.data_path, it))
                data["meta_data"].append(OrderedDict(niftii_obj.header))
                data["labels"] = 'labels'
                data['img'].append(niftii_obj.get_data())

        return data

    def set_info(self, info_path):
        """
        Set info for dataBase.
        :param info_path: info path
        :return: dictionary
        """
        ret_dic = {}
        with open(os.path.join(info_path, 'case_to_hash.json'), 'r') as f:
            ret_dic["case_to_hash"] = json.load(f)

        with open(os.path.join(info_path, 'train_test_list.json'), 'r') as f:
            ret_dic["train_test_list"] = json.load(f)

        return ret_dic

    def show_data(self, data):
        """
        Show data
        :param data : 3d matrix
        :return:
        """
        OrthoSlicer3D(data)
