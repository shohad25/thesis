import sys
import os
import glob
import numpy as np
import random
import argparse

from appcode.mri.k_space.files_info import files_info


def shuffle_data(base_dir, output_dir, tt='["train", "test"]', seed=123):
    """
    Given base train / test dir, this script shuffle examples and create 1 batch of data
    :param base_dir: basic dir
    :param output_dir: output dir
    :param tt: train / test
    :param seed: for pseudorandom initialization
    :return: None
    """

    random.seed(seed)

    tt = eval(tt)
    for t in tt:
        # Get all examples

        all_examples = glob.glob(os.path.join(base_dir, t, "*/*%s*" % files_info.keys()[0]))
        # Permute the list:
        random.shuffle(all_examples)

        # Now, write example according to all_examples order
        for (file_name, file_info) in files_info.iteritems():
            # Set output file name
            out_path = os.path.join(output_dir, t, "000000.%s.bin" % file_name)
            print "%s - Write %s into %s" % (t, file_name, out_path)
            with open(out_path, 'wb') as f_out:
                count_examples = 0
                # read each example and write it
                for example in all_examples:
                    base_input_path = os.path.dirname(example)
                    counter = os.path.basename(example).split('.')[0]
                    data = np.fromfile(os.path.join(base_input_path, "%s.%s.bin" % (counter, file_name)),
                                       dtype=file_info["dtype"])
                    f_out.write(data)

                    count_examples += 1
                    if np.mod(count_examples, 1000) == 0:
                        print "Done %d examples" % count_examples


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', required=True, type=str, help='Basic input directory of base data')

    parser.add_argument('--output_dir', required=True, type=str, help='Basic output directory for shuffle')

    parser.add_argument('--tt', required=False, default='["train", "test"]', type=str, help='Train / Test')

    parser.add_argument('--seed', required=False, default=123, type=int, help='Seed for pseudorandom initialization')

    args = parser.parse_args()

    # Run shuffle:
    shuffle_data(args.base_dir, args.output_dir, args.tt, args.seed)
