"""
dataCreator -
"""
import os
import sys
import numpy as np
from appcode.mri.k_space.utils import get_dummy_k_space_and_image
from appcode.mri.k_space.files_info import get_file_info
from common.files_IO.file_handler import FileHandler
import matplotlib.pyplot as plt
from common.viewers.imshow import imshow

# MAX_IM_VAL = 2.0**16 - 1

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
        self.base_out_path = os.path.join(output_path, 'base')
        self.axial_limits = axial_limits

    def create_examples(self, item='all', debug=False):
        """
        Data creation
        :param item: case name or all cases
        :return:
        """
        data_base = self.mri_data_base
        items = self.mri_data_base.items if item == "all" else [item]

        # For all cases in items
        for case in items:
            case_name = case.split('.')[0]
            print "Working on case: " + case_name

            # Set output path and create dir
            tt = self.mri_data_base.info["train_test_list"][case_name]
            out_path = os.path.join(self.base_out_path, tt, case_name)
            os.makedirs(out_path)
            counter = 0

            # Read source data and create k_space + dummy image
            source_data = data_base.get_source_data(case)
            image_3d = source_data['img'][0]

            # Normalize image
            norm_factor = 1.0 / image_3d.max() 
            image_3d = (image_3d * norm_factor).astype('float32')
            k_space_3d, dummy_image_3d = get_dummy_k_space_and_image(image_3d)
            meta_data = source_data['meta_data'][0]
            import pdb
            pdb.set_trace()
            print "TODO: SAVE NORM FACTOR"
            # Set image sizes
            w = image_3d[:,:,0].shape[0]
            h = image_3d[:,:,0].shape[1]

            # For each Z in axial limits, create masks and dump examples
            for z in range(self.axial_limits[0], self.axial_limits[1]+1):
                # Set ground truth
                k_space_2d_gt = k_space_3d[:, :, z]
                k_space_real_gt = k_space_2d_gt.real
                k_space_imag_gt = k_space_2d_gt.imag
                dummy_image_2d_gt = dummy_image_3d[:, :, z]

                # Subsample with factor = factor
                factor = 2
                for mask_type in range(0,2):
                    
                    # Create mask
                    mask = get_random_mask(w, h, factor=factor, start_line=mask_type)

                    # Get sub-sampled images, currently only on h axis
                    dummy_image_2d = get_subsample(dummy_image_2d_gt, mask, factor_h=factor, factor_w=1)
                    k_space_real = get_subsample(k_space_real_gt, mask, factor_h=factor, factor_w=1)
                    k_space_imag = get_subsample(k_space_imag_gt, mask, factor_h=factor, factor_w=1)

                    # Dump example
                    meta_data_to_write = self.create_meta_data(meta_data, case_name, z, factor)

                    dump_example(out_path, counter,
                                 dict(k_space_real_gt=k_space_real_gt, k_space_imag_gt=k_space_imag_gt,
                                      image_gt=dummy_image_2d_gt, mask=mask, meta_data=meta_data_to_write,
                                      k_space_real=k_space_real, k_space_imag=k_space_imag,
                                      image=dummy_image_2d), debug)
                    # Add to counter
                    counter += 1

    def create_meta_data(self, meta_data, case, axial_slice, factor):
        """
        Create meta data vector
        :param meta_data: from mri data base
        :param case: case name
        :param axial_slice: axial slice number
        :param factor: sub-sampling factor
        :return:
        """
        case_hash = np.float32(self.mri_data_base.info["case_to_hash"][case])
        bit_pix = np.float32(meta_data["bitpix"])
        return np.array([case_hash, axial_slice, bit_pix, np.float32(factor)], dtype=np.float32)


def dump_example(out_path, counter, data_all, debug=False):
    """
    write 1 example to disk
    :param out_path: output path
    :param counter: counter for count examples
    :param data_all: dictionary of all data types
    :param debug: show all data before dumping
    :return: None
    """

    if debug:
        show_example(out_path, counter, data_all)

    for (name, data) in data_all.iteritems():
        # Set file name
        file_name = set_file_name(name, counter)

        # Create file handler and write to file
        f_handler = FileHandler(os.path.join(out_path, file_name), get_file_info(name), "write")
        f_handler.write(data.transpose())


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


def get_subsample(image, mask, factor_h, factor_w):
    """
    Get sub-sample image according to mask
    :param mask: bool image
    :param image: original image
    :param factor_w: sub sampling factor on width
    :param factor_h: sub sampling factor on height
    :return: sub-sampled image
    """
    # Original size
    h = mask.shape[0]
    w = mask.shape[1]
    
    # New size
    h_new = int(np.floor(h / factor_h))
    w_new = int(np.floor(w / factor_w))

    # Sub-sample and reshape
    im_sub = image[np.where(mask==1)]
    im_sub_reshaped = im_sub.reshape(h_new, w_new)

    return im_sub_reshaped


def show_example(out_path, counter, data_all):
    """
    Show example - need to re-write every time we change the data
    :param out_path: output writing path
    :param counter: example number
    :param data_all: dictionary with all data
    """
    print "out_path: " + out_path
    print "counter: " + str(counter)

    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.set_size_inches(18.5, 10.5, forward=True)

    ax[0][0].set_title('Original Image')
    imshow(data_all["image_gt"], ax=ax[0][0], fig=fig)

    ax[0][1].set_title('SubSampled Image')
    imshow(data_all["image"], ax=ax[0][1], fig=fig)

    ax[1][0].set_title('Origianl real k-space')
    imshow(data_all["k_space_real_gt"], ax=ax[1][0], fig=fig)

    ax[1][1].set_title('SubSampled real k-space')
    imshow(data_all["k_space_real"], ax=ax[1][1], fig=fig)

    ax[2][0].set_title('Origianl imaginary k-space')
    imshow(data_all["k_space_imag_gt"], ax=ax[2][0], fig=fig)

    ax[2][1].set_title('SubSampled imaginary k-space')
    imshow(data_all["k_space_imag"], ax=ax[2][1], fig=fig)

    plt.waitforbuttonpress(timeout=-1)
    plt.close()