"""
dataCreator -
"""
import os
import numpy as np


class DataCreator:
    """
    DataCreator - Class of data creator object. Gets MriDataBase object and create example for training
    """

    def __init__(self, mri_data_base, output_path, axial_limits=np.array([10, 90])):
        """
        Constructor
        :type mri_data_base: MriDataBase
        :type axial_limits: numpy array
        :param mri_data_base: MriDataBase object
        :param output_path: base output path
        :param axial_limits: TBD
        :return:
        """
        self.mri_data_base = mri_data_base
        self.base_out_path = output_path
        self.axial_limits = axial_limits

    def get_random_mask(self, w, h, factor, start_line=0, keep_center=0):
        # TODO:: insert Randomness
        """
        Get random mask for k-space
        :param w: width
        :param h: height
        :param factor: sub-sampling factor
        :param start_line: bias
        :param keep_center: keeping line from the center. If 0, subsample uniformly,
                     Otherwise, keep is the center percentile [0,1]
        :return: numpy array - mask
        """
        mask = np.zeros((h, w))
        mask[range(start_line, h, factor), :] = 1

        if 0.0 < keep_center < 1.0:
            center_line = int(np.floor(h / 2))
            center_width = int(np.floor(0.5 * keep_center * h))
            mask[range(center_line-center_width, center_line+center_width, 1), :] = 1

        return mask


    def eval_mask(self):
        a = 1

    def dump_example(self):
        """
        
        :return:
        """

    def create_examples(self):
        a = 1
