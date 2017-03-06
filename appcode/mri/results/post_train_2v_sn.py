# !/home/ohadsh/Tools/anaconda/bin/python
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from appcode.mri.k_space.k_space_data_set import KspaceDataSet
from appcode.mri.k_space.utils import get_image_from_kspace, interpolated_missing_samples, zero_padding
from common.files_IO.file_handler import FileHandler
from common.viewers.imshow import imshow
file_names = ['image_gt', 'k_space_real_gt', 'k_space_imag_gt', 'mask', 'k_space_real', 'k_space_imag']
mini_batch = 50
from scipy import ndimage
from collections import defaultdict

base_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
# base_dir = '/sheard/Ohad/thesis/data/SchizData/SchizReg/train/2017_03_02_10_percent/shuffle'
with open(os.path.join(base_dir, "factors.json"), 'r') as f:
    data_factors = json.load(f)


def post_train_2v(data_dir, predict_paths, h=256, w=256, tt='test', show=False):
    """
    This function read predictions (dictionary) and compare it to the data
    :param data_dir: data main directory
    :param predict_paths: dictionary
    :param h: height
    :param w: width
    :param tt: train or test
    :param show: show flag
    :return:
    """
    #
    # mu_r = np.float32(data_factors['mean']['k_space_real'])
    # sigma_r = np.sqrt(np.float32(data_factors['variance']['k_space_real']))
    # # norm_r = lambda x: (x * sigma_r) + mu_r
    # norm_r = lambda x: (x - mu_r) / sigma_r
    #
    # mu_i = np.float32(data_factors['mean']['k_space_imag'])
    # sigma_i = np.sqrt(np.float32(data_factors['variance']['k_space_imag']))
    # # norm_i = lambda x: (x * sigma_i) + mu_i
    # norm_i = lambda x: (x - mu_i) / sigma_i

    method = 'bilinear'
    predict_info = {'width': w, 'height': h, 'channels': 1, 'dtype': 'float32'}
    interp_info = {'width': w, 'height': h, 'channels': 2, 'dtype': 'float32'}

    f_predict = defaultdict(dict)
    for (pred_name, pred_path) in predict_paths.iteritems():
        if pred_name != 'interp':
            f_predict[pred_name]['real'] = FileHandler(path=os.path.join(pred_path, "000000.predict_real.bin"),
                                                       info=predict_info, read_or_write='read', name=pred_name)
            f_predict[pred_name]['imag'] = FileHandler(path=os.path.join(pred_path, "000000.predict_imag.bin"),
                                                       info=predict_info, read_or_write='read', name=pred_name)

    f_interp_mc = FileHandler(path=predict_paths['interp'], info=interp_info, read_or_write='read', name='interp_mc')

    data_set = KspaceDataSet(data_dir, file_names, stack_size=50, shuffle=False)

    data_set_tt = getattr(data_set, tt)
    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.set_size_inches(18.5, 10.5, forward=True)
    
    while data_set_tt.epoch == 0:
        # Running over all data until epoch > 0
        data = data_set_tt.next_batch(mini_batch, norm=False)

        pred_real = {pred_name: pred_io['real'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}
        pred_imag = {pred_name: pred_io['imag'].read(n=mini_batch, reshaped=True) for (pred_name, pred_io) in f_predict.iteritems()}

        mc_interp = f_interp_mc.read(n=mini_batch, reshaped=True)

        real_p = {pred_name: pred_data for pred_name, pred_data in pred_real.iteritems()}
        imag_p = {pred_name: pred_data for pred_name, pred_data in pred_imag.iteritems()}

        real_interp = mc_interp[:,0,:,:]
        imag_interp = mc_interp[:,1,:,:]

        for i in range(0, real_interp.shape[0]):

            # Original image
            k_space_real_gt = data["k_space_real_gt"][i,:,:]
            k_space_imag_gt = data["k_space_imag_gt"][i,:,:]
            k_space_amp_gt = np.log(1+np.sqrt(k_space_real_gt**2 + k_space_imag_gt**2))
            org_image = get_image_from_kspace(k_space_real_gt,k_space_imag_gt)

            # Interpolation
            # mask = np.zeros_like(data["mask"][i,:,:])
            # mask[100:150,:] = 1.0

            # Apply low pass
            # sigma = 0.5
            # real_interp2 = ndimage.gaussian_filter(real_interp, sigma)
            # imag_interp2 = ndimage.gaussian_filter(imag_interp, sigma)
            # real_interp = real_interp * mask.T
            # imag_interp = imag_interp * mask.T
            rec_image_interp = get_image_from_kspace(real_interp, imag_interp)[i,:,:].T
            k_space_amp_interp = np.log(1+np.sqrt(real_interp**2 + imag_interp**2))[i,:,:].T

            # Network predicted model 1
            name_1 = real_p.keys()[0]
            # sigma = 1.0
            # Apply low pass
            # real_p[name_1] = ndimage.gaussian_filter(real_p[name_1], sigma)
            # imag_p[name_1] = ndimage.gaussian_filter(imag_p[name_1], sigma)

            # real_p[name_1] = real_p[name_1] * mask.T
            # imag_p[name_1] = imag_p[name_1] * mask.T
            rec_image_1 = get_image_from_kspace(real_p[name_1], imag_p[name_1])[i,:,:].T
            k_space_amp_predict_1 = np.log(1+np.sqrt(real_p[name_1]**2 + imag_p[name_1]**2))[i,:,:].T

            # Network predicted model 2
            name_2 = real_p.keys()[1]
            rec_image_2 = get_image_from_kspace(real_p[name_2], imag_p[name_2])[i,:,:].T
            k_space_amp_predict_2 = np.log(1+np.sqrt(real_p[name_2]**2 + imag_p[name_2]**2))[i,:,:].T

            ############ Original############
            ax[0][0].set_title('Original Image')
            ax[0][0].imshow(org_image, interpolation="none", cmap="gray")

            ax[1][0].set_title('Original K-space')
            ax[1][0].imshow(k_space_amp_gt, interpolation="none", cmap="gray")
            
            ########### Interpolated ############
            ax[0][1].set_title('Rec Image Interp:%s ' % method)
            ax[0][1].imshow(rec_image_interp, interpolation="none", cmap="gray")
            #
            ax[1][1].set_title('Interp K-space:%s ' % method)
            ax[1][1].imshow(k_space_amp_interp, interpolation="none", cmap="gray")

            ########### DNN 1 ############
            ax[0][2].set_title('DNN Reconstructed Image - ' + name_1)
            ax[0][2].imshow(rec_image_1, interpolation="none", cmap="gray")

            ax[1][2].set_title('DNN K-space _ ' + name_1)
            ax[1][2].imshow(k_space_amp_predict_1, interpolation="none", cmap="gray")

            ########### DNN 2 ############
            ax[0][3].set_title('DNN Reconstructed Image  - ' + name_2)
            ax[0][3].imshow(rec_image_2, interpolation="none", cmap="gray")

            ax[1][3].set_title('DNN K-space - ' + name_2)
            ax[1][3].imshow(k_space_amp_predict_2, interpolation="none", cmap="gray")

            plt.draw()

            plt.waitforbuttonpress(timeout=-1)

    plt.close()

if __name__ == '__main__':
    data_dir = '/home/ohadsh/work/data/SchizReg/24_05_2016/'
    #
    predict = {'ver1': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/singleNets/2017_03_05/predict/train/',
               'ver2' : '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/singleNets/2017_03_05/predict/train/',
               'interp': '/media/ohadsh/sheard/googleDrive/Master/runs/factor_2_phase/gan/singleNets/2017_03_05/predict/train/000000.interp.bin'
               }
    w = 256
    h = 256
    tt = 'train'
    show = False
    post_train_2v(data_dir, predict, h, w, tt, show)
