# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import copy
import sys
NII_SUFFIX = '.nii.gz'
SEG_SUFFIX = '_seg'
BRAIN_SUFFIX = '_brain'
CLASSES = [0,1,2,3]


def viewer(data_dir, num_of_cases=-1, suffixes=None, brain_only=True, view=None):
    """ Go over segmentation results and calc
    TPR, TNR, ACC, DICE score
    :param data_dir:
    :param num_of_cases:
    :param suffixes:
    :param brain_only: calculate on brain only
    :return:
    """
    if brain_only:
        seg_suffix_use = BRAIN_SUFFIX+SEG_SUFFIX
    else:
        seg_suffix_use = SEG_SUFFIX

    suffixes = eval(suffixes) if suffixes is not None else None

    view = eval(view) if view is not None else ['images', 'brains', 'segs']

    num_of_cases = 1000000000 if num_of_cases == -1 else num_of_cases
    sub_dirs = os.listdir(data_dir)
    PSNR = {suffix: [] for suffix in suffixes}
    MSE = {suffix: [] for suffix in suffixes}
    for sub_dir in sub_dirs:
        if sub_dirs.index(sub_dir) > num_of_cases - 1:
            break
        if not os.path.isdir(os.path.join(data_dir, sub_dir)):
            continue
        path_sub_dir = os.path.join(data_dir, sub_dir)
        all_files_in_dir = os.listdir(path_sub_dir)

        suffix_to_nifti_brain = {}
        suffix_to_nifti_seg = {}
        suffix_to_nifti_image = {}

        print("Working on - %s, number: (%d / %d)" % (sub_dir, len(MSE[MSE.keys()[0]]), num_of_cases))
        # Collect all relevant files for this case
        for file_type in all_files_in_dir:
            for suffix in suffixes:
                if file_type.endswith(sub_dir+suffix+seg_suffix_use+NII_SUFFIX):
                    suffix_to_nifti_seg[suffix] = nib.load(os.path.join(path_sub_dir, file_type))
                    brain_path = os.path.join(path_sub_dir, file_type.replace('_seg', ''))
                    suffix_to_nifti_brain[suffix] = nib.load(brain_path)
                    suffix_to_nifti_image[suffix] = nib.load(brain_path.replace('_brain', ''))

        if '' in suffix_to_nifti_seg.keys():
            mse = calc_statistics(suffix_to_nifti_seg, suffix_to_nifti_brain)
            print sub_dir
            for suffix in suffix_to_nifti_seg.keys():
                MSE[suffix].append(mse[suffix])

        args_to_view = {}
        if 'images' in view:
            args_to_view['images'] = copy.deepcopy(suffix_to_nifti_image)
        if 'brains' in view:
            args_to_view['brains'] = copy.deepcopy(suffix_to_nifti_brain)
        if 'segs' in view:
            args_to_view['segs'] = copy.deepcopy(suffix_to_nifti_seg)

        ret = view_samples(args_to_view)
        if ret == 0:
            return

    # Print results
    print('Results\n')
    for suffix in PSNR.keys():
        mse_array = np.array(MSE[suffix])
        psnr_mean = calc_psnr(mse_array).mean(axis=0)
        psnr_std = calc_psnr(mse_array).std(axis=0)
        print("Version - %s" % suffix)
        # Print total result
        print("PSNR  = %f, %f ||| %s , %s" % (psnr_mean.mean(), psnr_std.mean(), psnr_mean, psnr_std))


def calc_statistics(suffix_to_nifti, suffix_to_nifti_imag):
    """
    Get dictionary - suffix to nifti, return statistics
    :param suffix_to_nifti:
    :return:
    """
    ref_seg = suffix_to_nifti[''].get_data()
    reg_img = suffix_to_nifti_imag[''].get_data()
    mse = {}
    psnr = {}
    num_of_cls = len(CLASSES)
    for (ver_name, ver_nifti) in suffix_to_nifti_imag.iteritems():
        if ver_name == '':
            mse[ver_name] = num_of_cls*[0.0]
            psnr[ver_name] = num_of_cls*[0.0]
            continue
        data = ver_nifti.get_data()
        mse_temp = []

        for cls in CLASSES:
            mask = (ref_seg == cls).astype(np.int)
            mse_cls = np.mean(np.sum((mask * (data - reg_img))**2, axis=(0,1)))
            mse_temp.append(mse_cls)

        mse[ver_name] = mse_temp

    return mse


def calc_psnr(mse):
    max_val = 256
    psnr = 20*np.log10(max_val / np.sqrt(mse))
    return psnr


def view_samples(args_to_view):
    """
    :param args_to_view:
    :return:
    """
    suffixes = args_to_view[args_to_view.keys()[0]].keys()
    views = args_to_view.keys()

    # get data
    for (view, sufs) in args_to_view.iteritems():
        for suf in sufs.keys():
            args_to_view[view][suf] = args_to_view[view][suf].get_data()

    give_me_more = [True]
    val_to_ret = [1]
    save_example = [False]

    i = [1]

    def on_key(event):
        key = event.key
        if key == 'control' or key == 'alt' or key == 'ctrl+alt':
            return
        print 'you pressed %r' % key
        if key == ' ':
            i[0] += 1
            # call the draw_samples again with next images
        elif key == 'd':
            i[0] = i[0]
            save_example[0] = True
        elif key == 'p':
            i[0] = max(i[0] - 1, 1)
        if key == ']':
            i[0] += 10
            # call the draw_samples again with next images
        elif key == '[':
            i[0] = max(i[0] - 10, 1)
        elif key == 'q' or key == 'x':
            plt.close(fig)
            give_me_more[0] = False
            val_to_ret[0] = 1 if key == 'q' else 0
            return

    fig, ax = plt.subplots(nrows=len(views), ncols=len(suffixes))

    if len(views) == 1 and len(suffixes) == 1:
        ax = np.array(ax).reshape(1,1)
    else:
        if len(views) == 1:
            ax = ax.reshape(1, -1)
        if len(suffixes) == 1:
            ax = ax.reshape(-1, 1)

    # fig.tight_layout()
    fig.set_size_inches(18.5, 10.5, forward=True)

    while give_me_more[0]:
        for view in views:
            ax_col = views.index(view)
            for suffix in suffixes:
                ax_row = suffixes.index(suffix)
                # ax[ax_col][ax_row].set_title('%d' % i[0])
                ax[ax_col][ax_row].imshow(args_to_view[view][suffix][:, :, i[0]],
                                          interpolation="none", cmap="gray")
                ax[ax_col][ax_row].axis('off')
        plt.draw()

        fig.canvas.mpl_connect('key_press_event', on_key)
        sys.stdout.flush()
        if not plt.waitforbuttonpress():
            fig.tight_layout()
            pass

        if save_example[0]:
            print("Save example")

    print('Move to new case')
    return val_to_ret[0]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Go over segmentation results and calc statistics.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    parser.add_argument('--view', dest='view', default=None, type=str, help='views')
    parser.add_argument('--brain_only', dest='brain_only', default=True, type=bool, help='brain only')
    args = parser.parse_args()

    viewer(args.data_dir, args.num_of_cases, args.suffixes, args.brain_only, args.view)