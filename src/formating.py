import numpy as np


def makeSameSize(img1, img2):
    """
    This function makes it so that the two input images
    will have the same size, and set zeros to the areas of
    the images that are not used.
    """
    maxwidth = img1.shape[1]
    maxheight = img1.shape[0]
    if img2.shape[1] > maxwidth:
        maxwidth = img2.shape[1]
    if img2.shape[0] > maxheight:
        maxheight = img2.shape[0]
    newImg1 = np.zeros((maxheight, maxwidth, 4))
    newImg2 = np.zeros((maxheight, maxwidth, 4))
    newImg1[:img1.shape[0], :img1.shape[1], :] = img1
    newImg2[:img2.shape[0], :img2.shape[1], :] = img2
    return newImg1, newImg2
