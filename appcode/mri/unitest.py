import numpy as np

from common.orthoslicer import OrthoSlicer3D
from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.utils import get_dummy_k_space_and_image

a = MriDataBase('SchizReg')
data = a.get_source_data(a.items[0])

dat = data['img'][0]

dim1, dim2 = get_dummy_k_space_and_image(dat)
OrthoSlicer3D(np.log(0.1+np.abs(dim1)))
# OrthoSlicer3D(dat, aspect_ratio=np.array((2,1,1)))
