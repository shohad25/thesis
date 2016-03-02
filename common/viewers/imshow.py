import matplotlib.pyplot as plt


def imshow(image, colormap='gray'):
    # fig, ax = plt.subplot()
    plt.imshow(image, interpolation="none", cmap=plt.get_cmap(colormap))
    plt.colorbar()
    plt.show(block=False)