import numpy as np
import matplotlib.pyplot as plt
import sys


sys.path.append("/home/ohadsh/work/python/thesis/")

from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.utils import get_dummy_k_space_and_image
from common.viewers.imshow import imshow


to_show = True

data_source = MriDataBase('SchizReg')

# k-space creation and view
data_example = data_source.get_source_data(data_source.items[0])
dat = data_example['img'][0]
original_image = dat[:,:,50]
k_space, dummy_image = get_dummy_k_space_and_image(dat[:,:,50])

if to_show:

    # OrthoSlicer3D(np.log(0.1+np.abs(dim1)))
    # OrthoSlicer3D(dat, aspect_ratio=np.array((2,1,1)))
    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0][0].set_title('Original Image')
    imshow(original_image, ax=ax[0][0], fig=fig)

    ax[0][1].set_title('Log-k_space')
    imshow(np.log(1+np.abs(k_space)), ax=ax[0][1], fig=fig)

    ax[1][0].set_title('Dummy Image')
    imshow(dummy_image, ax=ax[1][0], fig=fig)

    ax[1][1].set_title('Diff')
    imshow(np.log(np.abs(original_image - dummy_image)), ax=ax[1][1], fig=fig)