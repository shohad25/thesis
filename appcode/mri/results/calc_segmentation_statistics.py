# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
import nibabel as nib
NII_SUFFIX = '.nii.gz'
SEG_SUFFIX = '_seg'


def calc_segmentation_statistics(data_dir, num_of_cases=-1, suffixes=None):
    """ Go over segmentation results and calc
    TPR, TNR, ACC, DICE score
    :param data_dir:
    :param num_of_cases:
    :param suffixes:
    :return:
    """
    suffixes = eval(suffixes) if suffixes is not None else None
    num_of_cases = 1000000000 if num_of_cases == -1 else num_of_cases
    sub_dirs = os.listdir(data_dir)
    TPR = {suffix: [] for suffix in suffixes}
    FPR = {suffix: [] for suffix in suffixes}
    DICE = {suffix: [] for suffix in suffixes}
    for sub_dir in sub_dirs:
        if sub_dirs.index(sub_dir) > num_of_cases-1 or not os.path.isdir(sub_dir):
            break
        path_sub_dir = os.path.join(data_dir, sub_dir)
        all_files_in_dir = os.listdir(path_sub_dir)
        suffix_to_nifti = {}

        print("Working on - %s" % sub_dir)
        # Collect all relevant files for this case
        for file_type in all_files_in_dir:
            for suffix in suffixes:
                if file_type.endswith(sub_dir+suffix+SEG_SUFFIX+NII_SUFFIX):
                    suffix_to_nifti[suffix] = nib.load(os.path.join(path_sub_dir, file_type))

        if '' in suffix_to_nifti.keys():
            tpr_case, fpr_case, dice_case = calc_statistics(suffix_to_nifti)
            for suffix in suffix_to_nifti.keys():
                TPR[suffix].append(tpr_case[suffix])
                FPR[suffix].append(fpr_case[suffix])
                DICE[suffix].append(dice_case[suffix])
    # Print results
    for suffix in TPR.keys():
        tpr_array = np.array(TPR[suffix])
        fpr_array = np.array(FPR[suffix])
        dice_array = np.array(DICE[suffix])

        tpr_mean = np.mean(tpr_array)
        fpr_mean = np.mean(fpr_array)
        dice_mean = np.mean(dice_array)

        tpr_std = np.std(tpr_array)
        fpr_std = np.std(fpr_array)
        dice_std = np.std(dice_array)

        print("Version - %s" % suffix)
        print("TPR = %f, %f" % (tpr_mean, tpr_std))
        print("FPR = %f, %f" % (fpr_mean, fpr_std))
        print("DICE = %f, %f" % (dice_mean, dice_std))


def calc_statistics(suffix_to_nifti):
    """
    Get dictionary - suffix to nifti, return statistics
    :param suffix_to_nifti:
    :return:
    """
    CLASS = 3
    ref = suffix_to_nifti[''].get_data()
    ref = ref == CLASS
    tpr = {}
    fpr = {}
    dice = {}
    for (ver_name, ver_nifti) in suffix_to_nifti.iteritems():
        if ver_name == '':
            tpr[ver_name] = 1.0
            fpr[ver_name] = 0.0
            dice[ver_name] = 1.0
            continue
        data = ver_nifti.get_data()
        data = data == CLASS
        tpr[ver_name] = float(np.sum(data == ref)) / len(data.ravel())
        fpr[ver_name] = float(np.sum(data != ref)) / len(data.ravel())
        dice[ver_name] = calc_dice(data, ref)
    return tpr, fpr, dice


def calc_dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Go over segmentation results and calc statistics.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    args = parser.parse_args()

    calc_segmentation_statistics(args.data_dir, args.num_of_cases, args.suffixes)