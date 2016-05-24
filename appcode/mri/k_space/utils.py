import numpy as np
from numpy.fft import *

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