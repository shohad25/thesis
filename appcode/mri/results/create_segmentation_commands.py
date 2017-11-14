# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import os
import argparse
NII_SUFFIX = '.nii.gz'
SEG_CMD = '$FSLDIR/bin/fast -n 3 -I 4 '
BET_CMD = '$FSLDIR/bin/bet'
BET_PARAMS = '-f 0.5 -g 0'


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
    cmd_file_out_path = os.path.join(data_dir, 'segmentations.sh')
    with open(cmd_file_out_path, 'w') as f:
        f.write('setFSL\n')
        cmds = []
        for sub_dir in sub_dirs:
            if sub_dirs.index(sub_dir) > num_of_cases - 1:
                break
            if not os.path.isdir(os.path.join(data_dir, sub_dir)):
                continue
            path_sub_dir = os.path.join(data_dir, sub_dir)
            all_files_in_dir = os.listdir(path_sub_dir)
            for file_type in all_files_in_dir:
                for suffix in suffixes:
                    if file_type.endswith(suffix+NII_SUFFIX):
                        brain_name = file_type.split(NII_SUFFIX)[0]+'_brain'+NII_SUFFIX
                        org_path = os.path.join(path_sub_dir, file_type)
                        brain_path = os.path.join(path_sub_dir, brain_name)
                        bet_cmd = "%s %s %s %s\n" % (BET_CMD, org_path, brain_path, BET_PARAMS)
                        seg_cmd = "%s %s\n" % (SEG_CMD, brain_path)
                        rm_cmd = "echo rm_%s; rm -rf %s/*pve* \n" % (file_type, path_sub_dir)
                        if bet_cmd not in cmds and seg_cmd not in cmds and rm_cmd not in cmds:
                            cmds.append('echo Working on %s\n' % file_type)
                            cmds.append(bet_cmd)
                            cmds.append(seg_cmd)
                            cmds.append(rm_cmd)

        f.writelines(cmds)
    print("Done...")
    print(cmd_file_out_path)
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Create sh script for segmentation commands.')
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='data directory')
    parser.add_argument('--num_of_cases', dest='num_of_cases', type=int, default=-1,  help='number of cases')
    parser.add_argument('--suffixes', dest='suffixes', default=None, type=str, help='suffixes')
    args = parser.parse_args()

    create_segmentation_commands(args.data_dir, args.num_of_cases, args.suffixes)