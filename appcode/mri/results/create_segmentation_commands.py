# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
NII_SUFFIX = '.nii.gz'


def create_segmentation_commands(data_dir, num_of_cases=-1, suffixes=None):
    """ Create sh script for segmentation commands
    :param data_dir:
    :param num_of_cases:
    :param suffixes:
    :return:
    """
    suffixes = eval(suffixes) if suffixes is not None else None
    num_of_cases = 1000000000 if num_of_cases == -1 else num_of_cases
    sub_dirs = os.listdir(data_dir)
    with open(os.path.join(data_dir, 'segmentations.sh'), 'w') as f:
        f.write('setFSL\n')
        cmds = []
        for sub_dir in sub_dirs:
            if sub_dirs.index(sub_dir) > num_of_cases - 1 or not os.path.isdir(sub_dir):
                break
            path_sub_dir = os.path.join(data_dir, sub_dir)
            all_files_in_dir = os.listdir(path_sub_dir)
            for file_type in all_files_in_dir:
                for suffix in suffixes:
                    if file_type.endswith(suffix+NII_SUFFIX):
                        cmd = '$FSLDIR/bin/fast -I 1 -v %s\n' % os.path.join(path_sub_dir, file_type)
                        if cmd not in cmds:
                            cmds.append('echo Working on %s\n' % file_type)
                            cmds.append(cmd)
        f.writelines(cmds)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create sh script for segmentation commands.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    args = parser.parse_args()

    create_segmentation_commands(args.data_dir, args.num_of_cases, args.suffixes)