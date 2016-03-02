import numpy as np

def get_dummy_k_space_and_image(img):
    """
    Get an MRI image and return k_space and dummy image
    :param img: 2d numpy.array or 3d if dim=3
    :return: k_space, img
    """

    k_space = np.zeros_like(img) + 0j
    dummy_img = np.zeros_like(img)

    if len(img.shape) == 2:
        # Create k-space from image, assuming this is the fully-sampled k-space.
        k_space = np.fft.fftshift(np.fft.fft2(img))

        # Reconstruct image
        dummy_img = np.fft.iff2(k_space).real()
    else:
        for i in range(0, img.shape[2]):
            # Create k-space from image, assuming this is the fully-sampled k-space.
            k_space[:,:,i] = np.fft.fftshift(np.fft.fft2(img[:,:,i]))

            # Reconstruct image
            dummy_img[:,:,i] = np.fft.ifft2(k_space[:,:,i]).real

    return k_space, dummy_img