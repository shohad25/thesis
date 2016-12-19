import numpy as np
from numpy.fft import *
from scipy.interpolate import griddata
from scipy.misc import imresize

EPS = 0.001

def get_dummy_k_space_and_image(img):
    """
    Get an MRI image and return k_space and dummy image
    :param img: 2d numpy.array or 3d if dim=3
    :return: k_space, img
    """

    k_space = (np.zeros_like(img) + 0j).astype('complex64')
    dummy_img = np.zeros_like(img)

    if len(img.shape) == 2:
        # Create k-space from image, assuming this is the fully-sampled k-space.
        # k_space = fftshift(fft2(img)).astype('complex64')
        k_space = fft2c(img).astype('complex64')

        # Clipping
        k_space.real[np.where(np.abs(k_space.real) < EPS)] = 0
        k_space.imag[np.where(np.abs(k_space.imag) < EPS)] = 0

        # Reconstruct image
        # dummy_img = np.abs(ifft2(k_space)).astype(np.int16)
        dummy_img = np.real(ifft2c(k_space)).astype(np.float32)
    else:
        for i in range(0, img.shape[2]):
            # Create k-space from image, assuming this is the fully-sampled k-space.
            k_space[:,:,i] = (fft2c(img[:,:,i])).astype('complex64')

            # Clipping
            k_space[:,:,i].real[np.where(np.abs(k_space[:,:,i].real) < EPS)] = 0
            k_space[:,:,i].imag[np.where(np.abs(k_space[:,:,i].imag) < EPS)] = 0

            # Reconstruct image
            # dummy_img[:,:,i] = np.abs(ifft2(k_space[:,:,i])).astype(np.int16)
            dummy_img[:,:,i] = np.abs(np.real(ifft2c(k_space[:,:,i])).astype(np.float32))
            dummy_img[:,:,i][np.where(np.abs(dummy_img[:,:,i]) < EPS)] = 0
            dummy_img[:,:,i][np.where(dummy_img[:,:,i] > 1)] = 1

    return k_space, dummy_img


def get_image_from_kspace(k_real, k_imag):
    """
    Return image from real and imaginary k-space values
    :param k_real:
    :param k_imag:
    :return:
    """
    # k_real = k_real.transpose(np.roll(np.arange(len(k_real.shape)), 0))
    # k_imag = k_real.transpose(np.roll(np.arange(len(k_imag.shape)), 0))
    k_space = (np.zeros_like(k_real) + 0j).astype('complex64')
    ret_img = np.zeros_like(k_real)

    if len(k_real.shape) == 2:
        # Create k-space from real and imaginary part
        k_space.real = k_real
        k_space.imag = k_imag

        # Clipping
        k_space.real[np.where(np.abs(k_space.real) < EPS)] = 0
        k_space.imag[np.where(np.abs(k_space.imag) < EPS)] = 0

        # Reconstruct image
        ret_img = np.real(ifft2c(k_space)).astype(np.float32)
    else:
        for i in range(0, k_real.shape[0]):
            # Create k-space from image, assuming this is the fully-sampled k-space.
            k_space[i,:,:] = (np.zeros_like(k_real[i,:,:]) + 0j).astype('complex64')
            k_space[i,:,:].real = k_real[i,:,:]
            k_space[i,:,:].imag = k_imag[i,:,:]

            # Clipping
            k_space[i,:,:].real[np.where(np.abs(k_space[i,:,:].real) < EPS)] = 0
            k_space[i,:,:].imag[np.where(np.abs(k_space[i,:,:].imag) < EPS)] = 0

            # Reconstruct image
            # dummy_img[:,:,i] = np.abs(ifft2(k_space[:,:,i])).astype(np.int16)
            ret_img[i,:,:] = np.abs(np.real(ifft2c(k_space[i,:,:])).astype(np.float32))
            ret_img[i,:,:][np.where(np.abs(ret_img[i,:,:]) < EPS)] = 0
            ret_img[i,:,:][np.where(ret_img[i,:,:] > 1)] = 1

    return ret_img


def fft2c(x):
    """
    res = fft2c(x)
    orthonormal forward 2D FFT
    (c) Michael Lustig 2005
    :param x: numpy array (image)
    :return: orthonormal 2D fft
    """
    norm_factor = 1.0 / np.sqrt(len(x.ravel()))
    ret = norm_factor * fftshift(fft2(ifftshift(x)))
    return ret


def ifft2c(x):
    """
    res = ifft2c(x)
    orthonormal forward 2D IFFT
    (c) Michael Lustig 2005
    :param x: numpy array (image)
    :return: orthonormal 2D ifft
    """
    norm_factor = np.sqrt(len(x.ravel()))
    ret = norm_factor * ifftshift(ifft2(fftshift(x)))
    return ret

def zero_padding(img_input, mask):
    """
    return the image with the same dims as img_input.
    put zero in mask
    :param img_input: input image
    :param mask: binary mask
    """

    return img_input * mask

def interpolated_missing_samples(img_input, dims_out, method):
    """
    return the image with the same dims as img_input.
    Interpolated missing samples
    :param img_input: input image
    :param dims_out: output dimension tuple
    :param method: method of interpolation
    """
    if len(img_input.shape) > 2:
        # For 3D volume
        ret = np.zeros_like(img_input)
        for i in range(0, img_input.shape[0]):
            ret[i,:,:] = imresize(arr=img_input[i,:,:], size=dims_out, interp=method, mode=None)
    else:
        ret = imresize(arr=img_input, size=dims_out, interp=method, mode=None)
    return ret
        
