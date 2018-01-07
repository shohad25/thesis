# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
import nibabel as nib
from skimage.measure import compare_ssim as ssim
NII_SUFFIX = '.nii.gz'
SEG_SUFFIX = '_seg'
BRAIN_SUFFIX = '_brain'
CLASSES = [0]


def calc_masked_psnr(data_dir, num_of_cases=-1, suffixes=None, brain_only=True):
    """ Go over segmentation results and calc
    TPR, TNR, ACC, DICE score
    :param data_dir:
    :param num_of_cases:
    :param suffixes:
    :param brain_only: calculate on brain only
    :return:
    """
    seg_suffix_use = SEG_SUFFIX

    suffixes = eval(suffixes) if suffixes is not None else None
    num_of_cases = 1000000000 if num_of_cases == -1 else num_of_cases
    sub_dirs = os.listdir(data_dir)
    PSNR = {suffix: [] for suffix in suffixes}
    MSE = {suffix: [] for suffix in suffixes}
    SSIM = {suffix: [] for suffix in suffixes}
    for sub_dir in sub_dirs:
        if sub_dirs.index(sub_dir) > num_of_cases - 1:
            break
        if not os.path.isdir(os.path.join(data_dir, sub_dir)):
            continue
        path_sub_dir = os.path.join(data_dir, sub_dir)
        all_files_in_dir = os.listdir(path_sub_dir)
        suffix_to_nifti = {}

        print("Working on - %s, number: (%d / %d)" % (sub_dir, len(MSE[MSE.keys()[0]]), num_of_cases))
        # Collect all relevant files for this case
        for file_type in all_files_in_dir:
            for suffix in suffixes:
                if file_type.endswith(sub_dir+suffix+NII_SUFFIX):
                    suffix_to_nifti[suffix] = nib.load(os.path.join(path_sub_dir, file_type))

        if '' in suffix_to_nifti.keys():
            mse, ssim_ = calc_statistics(suffix_to_nifti)
            print sub_dir
            for suffix in suffix_to_nifti.keys():
                MSE[suffix].append(mse[suffix])
                SSIM[suffix].append(ssim_[suffix])

    # Print results
    print('Results\n')
    for suffix in PSNR.keys():
        # import pdb
        # pdb.set_trace()
        mse_array = np.array(MSE[suffix])
        arg_min = np.argmax(mse_array, axis=0)
        psnr_mean = calc_psnr(mse_array).mean(axis=0)
        psnr_std = calc_psnr(mse_array).std(axis=0)

        ssim_array = np.array(SSIM[suffix])
        ssim_mean = ssim_array.mean(axis=0)
        ssim_std = ssim_array.std(axis=0)

        print("Version - %s" % suffix)
        # Print total result
        print("PSNR  = %f [dB], %f [dB]" % (psnr_mean.mean(), psnr_std.mean()))
        print("SSIM  = %f, %f\n" % (ssim_mean.mean(), ssim_std.mean()))


def calc_statistics(suffix_to_nifti):
    """
    Get dictionary - suffix to nifti, return statistics
    :param suffix_to_nifti:
    :return:
    """
    reg_img = suffix_to_nifti[''].get_data()
    mse = {}
    ssim_ = {}
    num_of_cls = len(CLASSES)
    for (ver_name, ver_nifti) in suffix_to_nifti.iteritems():
        if ver_name == '':
            mse[ver_name] = num_of_cls*[0.0]
            ssim_[ver_name] = num_of_cls*[1.0]
            continue
        data = ver_nifti.get_data()
        mse_temp = []
        mse_cls = np.mean(np.sum(((data - reg_img))**2, axis=(0,1)))
        mse_temp.append(mse_cls)

        ssim_temp = ssim(X=reg_img, Y=data, data_range=data.max() - data.min())
        mse[ver_name] = mse_temp
        ssim_[ver_name] = ssim_temp

    return mse, ssim_


def calc_psnr(mse):
    max_val = 256
    psnr = 20*np.log10(max_val / np.sqrt(mse))
    return psnr


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Go over segmentation results and calc statistics.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    parser.add_argument('--brain_only', dest='brain_only', default=True, type=bool, help='brain only')
    args = parser.parse_args()

    calc_masked_psnr(args.data_dir, args.num_of_cases, args.suffixes, args.brain_only)