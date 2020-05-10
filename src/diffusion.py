import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt


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


def pre_diffuse(u, mask=None, met='e', rand='n', alpha=0.24, itr=50, h=0, D=1):
    u1 = np.copy(u)  # u1 for returned img, u for diriclet conditions

    # apply transformation to whole image if nothing is specified
    if mask is None:
        mask = np.ones(u.shape[:2])

    # Ensure mask is boolean and find where it is true
    mask = mask.astype(bool)
    maskCords = np.argwhere(mask)

    # Generate view of image for faster processing
    top = np.amin(maskCords[:, 0])
    bottom = np.amax(maskCords[:, 0]) + 1
    left = np.amin(maskCords[:, 1])
    right = np.amax(maskCords[:, 1]) + 1

    view = u[top:bottom, left:right]
    new_view = u1[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    # Clear borders of the mask to prevent wrapping
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    orig_mask = np.copy(mask)

    # Creates diffrent views for equation
    t_mask = np.roll(mask, -1, axis=0)
    b_mask = np.roll(mask, 1, axis=0)
    l_mask = np.roll(mask, -1, axis=1)
    r_mask = np.roll(mask, 1, axis=1)
    if met == 'e':    
        for i in range(itr):
            laplace = (
                new_view[t_mask] +
                new_view[b_mask] +
                new_view[l_mask] +
                new_view[r_mask] -
                4 * new_view[mask]
            )
            new_view = explicit(new_view, mask, laplace, alpha, h, D)
        
        if rand == 'd':
            new_view[~mask] = view[~mask]
    
    elif met == 'i':
        shape = view.shape

        # Vectorize image and mask
        view = view.ravel()
        new_view = new_view.ravel()
        mask = mask.ravel()

        if len(shape) == 3:
            view = view.reshape(shape[0] * shape[1], 3)
            new_view = new_view.reshape(shape[0] * shape[1], 3)

        size = view.shape[0]

        n_upperdiag = np.concatenate((
            np.zeros(shape[1]),
            -alpha * np.ones(size - shape[1])
            ))
        upperdiag = np.concatenate(([0, 0], -alpha * np.ones(size - 2)))
        centerdiag = np.concatenate((
                                        [1],
                                        (1 + 4 * alpha) * np.ones(size - 2),
                                        [1]
                                    ))
        lowerdiag = np.concatenate((-alpha * np.ones(size - 2), [0, 0]))
        n_lowerdiag = np.concatenate((
            -alpha * np.ones(size - shape[1]),
            np.zeros(shape[1])
            ))
        diag = np.array([
            n_upperdiag,
            upperdiag,
            centerdiag,
            lowerdiag,
            n_lowerdiag
            ])

        sparse = spdiags(
            diag,
            [shape[1], 1, 0, -1, -shape[1]],
            size,
            size
            ).tocsc()

        for i in range(itr):
            new_view = implicit(sparse, new_view, shape)

        new_view = new_view.reshape(shape)
        view = view.reshape(shape)
        mask = mask.reshape(shape[:2])

        
        # Diriclet conditions. Boundry should be returned to
        # original values (asumed known value)
        if rand == 'd':
            new_view[~mask] = view[~mask]

 

        u1[top:bottom, left:right] = new_view

    return u1


def explicit(u, mask, laplace=0, alpha=0.24, h=0, D=1):

    if isinstance(D, np.ndarray):
        u[mask] += laplace * alpha * D[mask]
    else:
        u[mask] += laplace * alpha

    # Neumann boundary condition du/dt = 0
    u[0, :] = u[1, :]
    u[:, 0] = u[:, 1]
    u[-1, :] = u[-2, :]
    u[:, -1] = u[:, -2]

    # only apply h if relevant, else rounding error occurs
    if isinstance(h, np.ndarray):
        return (u - h)

    return u

# currently only supported on blurring and inpaint.
def implicit(sparse, u, shape):
    u = spsolve(sparse, u)
    u[0::shape[1]] = u[1::shape[1]]
    u[shape[1]-1::shape[1]] = u[shape[1]-2::shape[1]]
    return u
