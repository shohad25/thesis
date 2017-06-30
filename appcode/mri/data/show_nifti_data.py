#!/usr/bin/python
import os
import argparse
import nibabel as nib
from common.viewers.orthoslicer import OrthoSlicer3D


def show_nifti_data(data_path):
    nii = nib.load(data_path)
    print nii.header
    print nii.shape
    OrthoSlicer3D(nii.get_data().squeeze())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Nifti data viewer.')
    parser.add_argument('--data_path', dest='data_path', default='', type=str, help='nifti data path')
    args = parser.parse_args()
    show_nifti_data(os.path.abspath(args.data_path))
