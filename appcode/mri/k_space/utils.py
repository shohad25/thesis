import numpy as np

EPS = 0.01

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
        k_space = np.fft.fftshift(np.fft.fft2(img)).astype('complex64')

        # Clipping
        k_space.real[np.where(np.abs(k_space.real) < EPS)] = EPS
        k_space.imag[np.where(np.abs(k_space.imag) < EPS)] = EPS

        # Reconstruct image
        dummy_img = np.abs(np.fft.ifft2(k_space)).astype(np.int16)
    else:
        for i in range(0, img.shape[2]):
            # Create k-space from image, assuming this is the fully-sampled k-space.
            k_space[:,:,i] = (np.fft.fftshift(np.fft.fft2(img[:,:,i]))).astype('complex64')

            # Clipping
            k_space[:,:,i].real[np.where(np.abs(k_space[:,:,i].real) < EPS)] = EPS
            k_space[:,:,i].imag[np.where(np.abs(k_space[:,:,i].imag) < EPS)] = EPS

            # Reconstruct image
            dummy_img[:,:,i] = np.abs(np.fft.ifft2(k_space[:,:,i])).astype(np.int16)

    return k_space, dummy_img