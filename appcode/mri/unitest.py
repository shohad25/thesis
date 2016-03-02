from appcode.mri.data.mri_data_base import MriDataBase
from appcode.mri.k_space.data_creator import DataCreator
from appcode.mri.k_space.utils import get_dummy_k_space_and_image

data_source = MriDataBase('SchizReg')
# data_example = data_example.get_source_data(data_example.items[0])

# dat = data_example['img'][0]
# dim1, dim2 = get_dummy_k_space_and_image(dat)
# OrthoSlicer3D(np.log(0.1+np.abs(dim1)))
# OrthoSlicer3D(dat, aspect_ratio=np.array((2,1,1)))

data_creator = DataCreator(data_source, "here")
