import numpy as np


def diffuse(u, alpha=0.24, h=0, D=1., rand='n', dr=0):
    """
    The general diffusion equation, used to
    diffuse a one-channel-float[0, 1]-image.
    """
    u1 = -4*u
    u1[0:-1, :] += u[1:, :]
    u1[1:, :] += u[0:-1, :]
    u1[:, 0:-1] += u[:, 1:]
    u1[:, 1:] += u[:, 0:-1]
    # Randbetingelser:
    if rand == 'n':
        # Neumann condition
        u1[0, :] += u[0, :]
        u1[:, 0] += u[:, 0]
        u1[-1, :] += u[-1, :]
        u1[:, -1] += u[:, -1]
    else:
        u1[0, :] += dr
        u1[:, 0] += dr
        u1[-1, :] += dr
        u1[:, -1] += dr
    return (u1 * D * alpha + h + u)


def gY(img):
    """
    Gradient Y component of an image.
    """
    img1 = np.zeros(img.shape)
    img1[1:-1, :] = (img[2:, :] - img[:-2, :])/2
    img1[0, :] = img[1, :]-img[0, :]
    img1[-1, :] = img[-1, :]-img[-2, :]
    return img1


def gX(img):
    """
    Gradient X component of an image.
    """
    img1 = np.zeros(img.shape)
    img1[:, 1:-1] = (img[:, 2:] - img[:, :-2])/2
    img1[:, 0] = img[:, 1]-img[:, 0]
    img1[:, -1] = img[:, -1]-img[:, -2]
    return img1


def D_Image(img, k):
    timg = np.zeros(img.shape)
    for i in range(timg.shape[2]):
        timg[:, :, i] = 1. / (1 + k * (gX(img)**2 + gY(img)**2))
    return timg
