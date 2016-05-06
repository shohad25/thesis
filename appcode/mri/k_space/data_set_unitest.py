import numpy as np
import sys
import matplotlib.pyplot as plt


from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from common.viewers.imshow import imshow

to_show = True

# k space data set
base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/06_05_2016/shuffle/'


file_names = ['k_space_real', 'k_space_imag', 'mask', 'meta_data', 'image']
data_set = KspaceDataSet(base_dir, file_names, stack_size=50)

batch = data_set.train.next_batch(data_set.train.N_MAX)

if to_show:
    fig, ax = plt.subplots(nrows=2, ncols=2)

    for i in range(0, 50):
        ax[0][0].set_title('mask')
        imshow(batch["mask"][i,:,:], ax=ax[0][0], fig=fig)

        ax[0][1].set_title('Image')
        imshow(batch["image"][i,:,:], ax=ax[0][1], fig=fig)

        ax[1][0].set_title('Log-k_space real')
        imshow(np.log(1+np.abs(batch["k_space_real"][i,:,:])), ax=ax[1][0], fig=fig)

        ax[1][1].set_title('Log-k_space imaginary')
        imshow(np.log(1+np.abs(batch["k_space_imag"][i,:,:])), ax=ax[1][1], fig=fig)

        # ax[1][1].set_title('meta_data')
        # imshow(batch["meta_data"][:,i], ax=ax[1][1], fig=fig)

        plt.waitforbuttonpress(timeout=-1)