#!/home/ohadsh/anaconda2/bin/python
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.data_creator import get_random_gaussian_mask, get_rv_mask, get_random_mask
from appcode.mri.k_space.utils import get_image_from_kspace
import matplotlib.pyplot as plt
import numpy as np

base_dir = '/media/ohadsh/Data/ohadsh/work/data/T1/sagittal/'
# base_dir = '/media/ohadsh/sheard/Ohad/thesis/data/SchizData/SchizReg/train/24_05_2016/shuffle/'
file_names = ['k_space_real_gt', 'k_space_imag_gt']
mini_batch = 50
tt = 'train'
data_set = KspaceDataSet(base_dir, file_names, stack_size=50, shuffle=False, data_base='IXI_T1')
data_set_tt = getattr(data_set, tt)

IMG_FORMAT = 'png'
DPI = 250

fig, ax = plt.subplots(nrows=1, ncols=4)
fig.set_size_inches(18.5, 10.5, forward=True)

# mask = get_random_gaussian_mask(im_shape=(256, 256), peak_probability=0.7, std=60.0, keep_center=0.1, seed=0)
mask = get_rv_mask(mask_main_dir='/media/ohadsh/Data/ohadsh/work/matlab/thesis/', factor=4)
# mask = get_random_mask(w=256, h=256, factor=2, keep_center=0.1)
for i in range(3,100):

    data = data_set_tt.next_batch(mini_batch, norm=False)
    k_space_real_gt = data["k_space_real_gt"][i,:,:]
    k_space_imag_gt = data["k_space_imag_gt"][i,:,:]

    org_image = get_image_from_kspace(k_space_real_gt, k_space_imag_gt)

    reduction = np.sum(mask) / float(mask.ravel().shape[0])
    print reduction
    zero_padding_image = get_image_from_kspace(k_space_real_gt*mask, k_space_imag_gt*mask)


    ax[0].set_title('Original Image')
    ax[0].imshow(org_image, interpolation="none", cmap="gray")
    ax[0].axis('off')

    ax[1].set_title('Zero padded')
    ax[1].imshow(zero_padding_image, interpolation="none", cmap="gray")
    ax[1].axis('off')

    ax[2].set_title('Mask - reduction=%f' % reduction)
    # k_space_amp = np.log(1 + np.sqrt(k_space_imag_gt ** 2))
    # k_space_amp = np.log(1 + np.sqrt((k_space_imag_gt * mask) ** 2))
    # ax[2].imshow(k_space_amp, interpolation="none", cmap="gray")
    addi = 0.000001

    temp = mask*np.abs(k_space_real_gt + 0j*k_space_imag_gt)
    temp[temp==0] = 0.0000001
    # k_space_amp = np.log(addi + np.sqrt((k_space_real_gt*mask) ** 2 + (k_space_imag_gt*mask) ** 2))
    k_space_amp = np.log(addi+temp)
    ax[2].imshow(k_space_amp, interpolation="none", cmap="gray")

    ax[2].axis('off')

    ax[3].set_title('K-space')
    # k_space_amp = np.log(addi + np.sqrt(k_space_real_gt ** 2 + k_space_imag_gt ** 2))
    temp = np.abs(k_space_real_gt + 0j*k_space_imag_gt)
    k_space_amp = np.log(addi+temp)
    # k_space_amp = np.log(1 + np.sqrt(k_space_real_gt ** 2))
    # k_space_amp = np.log(1 + np.sqrt((k_space_real_gt) ** 2))
    # ax[3].imshow(k_space_amp, interpolation="none", cmap="gray", vmin=k_space_amp.min(), vmax=k_space_amp.max())
    ax[3].imshow(mask, interpolation="none", cmap="gray")
    ax[3].axis('off')

    plt.draw()
    # # fig.tight_layout()
    # # Save
    # extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig('/tmp/example_original.png', bbox_inches=extent, format=IMG_FORMAT, dpi=DPI)
    # extent = ax[1].get_window_extent().transforme(dfig.dpi_scale_trans.inverted())
    # fig.savefig('/tmp/example_zeroPadding.png', bbox_inches=extent, format=IMG_FORMAT, dpi=DPI)
    
    extent = ax[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('/tmp/k_space_with_mask.png', bbox_inches=extent, format=IMG_FORMAT, dpi=DPI)
    extent = ax[3].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('/tmp/k_space.png', bbox_inches=extent, format=IMG_FORMAT, dpi=DPI)

    plt.waitforbuttonpress(timeout=0)
    exit(1)