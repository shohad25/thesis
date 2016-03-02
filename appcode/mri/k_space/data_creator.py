"""
dataCreator -
"""
import os
import sys
import numpy as np
from appcode.mri.k_space.utils import get_dummy_k_space_and_image
from appcode.mri.k_space.files_info import get_file_info
from common.files_IO.file_handler import FileHandler

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

    def create_examples(self, item='all'):
        """
        Data creation
        :param item: case name or all cases
        :return:
        """
        data_base = self.mri_data_base
        items = self.mri_data_base.items if item == "all" else [item]

        # For all cases in items
        for case in items:
            # Set output path and create dir
            out_path = os.path.join(self.base_out_path, case.split('.')[0])
            os.mkdir(out_path)
            counter = 0

            # Read source data and create k_space + dummy image
            source_data = data_base.get_source_data(case)
            image_3d = source_data['img'][0]
            k_space_3d, dummy_image_3d = get_dummy_k_space_and_image(image_3d)
            meta_data = source_data['meta_data'][0]

            w = image_3d[:,:,0].shape[0]
            h = image_3d[:,:,0].shape[1]

            # For each Z in axial limits, create masks and dump examples
            for z in range(self.axial_limits[0], self.axial_limits[1]+1):
                k_space_2d = image_3d[:, :, z]
                dummy_image_2d = dummy_image_3d[:, :, z]

                for mask_type in range(0,2):
                    # Create mask
                    mask = get_random_mask(w, h, factor=2, start_line=mask_type)

                    # Dump example
                    dump_example(out_path, counter,
                                 dict(k_space_real=k_space_2d, k_space_imag=k_space_2d,
                                      image=dummy_image_2d, mask=mask, meta_data=create_meta_data(meta_data)))
                    # Add to counter
                    counter += 1


def dump_example(out_path, counter, data_all):

    for (name, data) in data_all.iteritems():
        # Set file name
        file_name = set_file_name(name, counter)

        # Create file handler and write to file
        f_handler = FileHandler(os.path.join(out_path, file_name), get_file_info(name), "write")
        f_handler.write(data)


def set_file_name(name, counter):
    """
    Set file name according to counter
    :param name: basic name
    :param counter: counter - integer
    :return: string: for example - "0000123.features.bin"
    """
    base_number = "00000000"
    str_counter = str(counter)
    len_base = len(base_number)
    len_count = len(str_counter)
    assert(len_base > len_count)
    file_number = base_number[0:len_base - len_count] + str_counter
    return "%s.%s.bin" % (file_number, name)


def get_random_mask(w, h, factor, start_line=0, keep_center=0):
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
    mask = np.zeros((h, w), dtype=np.uint8)

    # Add lines to sample, according to factor and start line
    mask[range(start_line, h, factor), :] = 1

    # Keep lines of center of k-space (most of the energy)
    if 0.0 < keep_center < 1.0:
        center_line = int(np.floor(h / 2))
        center_width = int(np.floor(0.5 * keep_center * h))
        mask[range(center_line-center_width, center_line + center_width, 1), :] = 1
    return mask


def create_meta_data(meta_data):
    return np.array([0], dtype=np.float32)