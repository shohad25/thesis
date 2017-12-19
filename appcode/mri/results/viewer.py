# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
import copy
import sys
from PIL import Image
NII_SUFFIX = '.nii.gz'
SEG_SUFFIX = '_seg'
BRAIN_SUFFIX = '_brain'
IMG_FORMAT = 'tif'
CLASSES = [0,1,2,3]


class Viewer(object):
    def __init__(self, data_dir, output_dir='/tmp/', num_of_cases=-1, suffixes=None, brain_only=True, views=None, mask=None):
        """
        Viewer for post training, compare examples and dump images for paper
        :param data_dir:
        :param num_of_cases:
        :param suffixes:
        :param brain_only:
        :param views:
        """
        self.data_dir = data_dir
        self.num_of_cases = 1000 if num_of_cases == -1 else num_of_cases
        self.suffixes = eval(suffixes) if suffixes is not None else None
        self.brain_only = brain_only
        self.views = eval(views) if views is not None else ['images', 'brains', 'segs']
        self.seg_suffix_use = BRAIN_SUFFIX+SEG_SUFFIX if brain_only else SEG_SUFFIX
        self.sub_dirs = os.listdir(data_dir)
        self.output_dir = output_dir
        self.counter = 0
        self.data = None
        self.mask = eval(mask) if mask is not None else None

    def show(self, case_name=None):
        """
        Show function
        :param case_name:
        :return:
        """
        self.counter = 1
        for sub_dir in self.sub_dirs:
            if self.sub_dirs.index(sub_dir) > self.num_of_cases - 1:
                break
            if not os.path.isdir(os.path.join(self.data_dir, sub_dir)):
                continue

            # Load one case and store it in self.data
            self.load_one_case(sub_dir)

            # View one case
            ret = self.view_case(case_name=sub_dir)

            # Exit condition
            if ret == 0:
                return

            self.counter += 1

    def load_one_case(self, sub_dir):
        """
        Load one case
        :param sub_dir: 
        :return: 
        """
        path_sub_dir = os.path.join(self.data_dir, sub_dir)
        all_files_in_dir = os.listdir(path_sub_dir)

        suffix_to_nifti_brain = {}
        suffix_to_nifti_seg = {}
        suffix_to_nifti_image = {}

        print("Working on - %s, number: (%d / %d)" % (sub_dir, self.counter, self.num_of_cases))
        # Collect all relevant files for this case
        for file_type in all_files_in_dir:
            for suffix in self.suffixes:
                if file_type.endswith(sub_dir+suffix+self.seg_suffix_use+NII_SUFFIX):
                    suffix_to_nifti_seg[suffix] = nib.load(os.path.join(path_sub_dir, file_type)).get_data()
                    brain_path = os.path.join(path_sub_dir, file_type.replace('_seg', ''))
                    suffix_to_nifti_brain[suffix] = nib.load(brain_path).get_data()
                    suffix_to_nifti_image[suffix] = nib.load(brain_path.replace('_brain', '')).get_data()

        self.data = {}
        if 'images' in self.views:
            self.data['images'] = copy.deepcopy(suffix_to_nifti_image)
        if 'brains' in self.views:
            self.data['brains'] = copy.deepcopy(suffix_to_nifti_brain)
        if 'segs' in self.views:
            self.data['segs'] = copy.deepcopy(suffix_to_nifti_seg)

        # Slice according to given mask
        if self.mask is not None:
            for view in self.data.keys():
                for suffix in self.data[view].keys():
                    self.data[view][suffix] = \
                        self.data[view][suffix][self.mask['y1']: self.mask['y2'], self.mask['x1']: self.mask['x2'], :]

    def dump_example(self, case_name, view, suffix, i):
        if not os.path.exists(os.path.join(self.output_dir, case_name)):
            os.makedirs(os.path.join(self.output_dir, case_name))
        out_file = os.path.join(self.output_dir, case_name, '%s_%s_%s_%d.%s' %
                                (case_name, view, suffix, i[0], IMG_FORMAT))
        data = self.data[view][suffix][:, :, i[0]]
        formatted = (data * 65535 / np.max(data)).astype('uint16')
        # formatted = (data * 255 / np.max(data)).astype('uint8')
        # if suffix == '':
        #     self.idx_zeros = np.where(formatted == 0)
        # else:
        #     formatted[self.idx_zeros] = 0

        im = Image.fromarray(formatted)
        with open(out_file, 'w') as f:
            im.save(f)

    def view_case(self, case_name):
        """
        Show one case stored in self.data
        """
        suffixes = self.data[self.data.keys()[0]].keys()
        views = self.data.keys()

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
                    ax[ax_col][ax_row].imshow(self.data[view][suffix][:, :, i[0]],
                                              interpolation="none", cmap="gray")
                    ax[ax_col][ax_row].axis('off')
            plt.draw()

            fig.canvas.mpl_connect('key_press_event', on_key)
            sys.stdout.flush()
            if not plt.waitforbuttonpress():
                fig.tight_layout()
                pass

            if save_example[0]:
                save_example[0] = False
                print("Save example")
                for view in views:
                    for suffix in suffixes:
                        self.dump_example(case_name, view, suffix, i)

        print('Move to new case')
        return val_to_ret[0]


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Go over segmentation results and calc statistics.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--output_dir', dest='output_dir', type=str, default='/tmp/', help='output directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    parser.add_argument('--views', dest='views', default=None, type=str, help='views')
    parser.add_argument('--brain_only', dest='brain_only', default=True, type=bool, help='brain only')
    parser.add_argument('--mask', dest='mask', default=None, type=str, help='mask')
    args = parser.parse_args()

    viewer = Viewer(data_dir=args.data_dir, output_dir=args.output_dir, num_of_cases=args.num_of_cases,
                    suffixes=args.suffixes, brain_only=args.brain_only, views=args.views, mask=args.mask)
    viewer.show()
