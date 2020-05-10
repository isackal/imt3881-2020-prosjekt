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

    # Creates diffrent views for equation
    t_mask = np.roll(mask, -1, axis=0)
    b_mask = np.roll(mask, 1, axis=0)
    l_mask = np.roll(mask, -1, axis=1)
    r_mask = np.roll(mask, 1, axis=1)

    for i in range(itr):
        """
        TO BE TESTED:
        cloning
        Contrast
        kantBevGlatting
        """
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

    return u1


def explicit(u, mask, laplace=0, alpha=0.24, h=0, D=1):
    if isinstance(D, int):
        u[mask] += laplace * alpha
    else:
        u[mask] += laplace * alpha * D[mask]

    # Neumann boundary condition du/dt = 0
    u[0, :] = u[1, :]
    u[:, 0] = u[:, 1]
    u[-1, :] = u[-2, :]
    u[:, -1] = u[:, -2]

    # only apply h if relevant, else rounding error occurs
    if isinstance(h, int):
        return u
    return (u - h)
