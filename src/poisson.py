import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import bicgstab
import imageio


def explTransform(views, alpha):
    """
    Creates a laplace for an explisit transformation

    Paramters
    ---------
    views : tuple
        5 np.ndarrays around region of transformation

    alpha : float
        delta_t / delta_x**2

    Returns
    -------
    np.ndarray
        laplace transformation
    """
    laplace = (
                views[0] +
                views[1] +
                views[2] +
                views[3] -
                4 * views[4]
                )
    return alpha*laplace


def explisitt(img, depth, mask, alpha):
    """
    Runs an explisit diffusion of an image

    Alpha should remain below 0.24 to prevent numeric
    instablilty.

    Paramters
    ---------
    img : np.ndarray
        Image to be diffused

    depth : int
        Number of itterations to run the diffusion

    mask : np.ndarray
        Boolean array for where to run diffusion.

    alpha : float
        delta_t / delta_x**2

    Returns
    -------
    np.ndarray
        Diffused image
    """
    new_img = img.astype(float) / 255
    img = img.astype(float) / 255
    mask = mask.astype(bool)
    maskCords = np.argwhere(mask)

    # Generate view of image around inpaint region
    top = np.amin(maskCords[:, 0])
    bottom = np.amax(maskCords[:, 0]) + 1
    left = np.amin(maskCords[:, 1])
    right = np.amax(maskCords[:, 1]) + 1

    view = img[top:bottom, left:right]
    new_view = new_img[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    # Do not solve equation on boundry
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    # Creates diffrent views for equation
    t_mask = np.roll(mask, -1, axis=0)
    b_mask = np.roll(mask, 1, axis=0)
    l_mask = np.roll(mask, -1, axis=1)
    r_mask = np.roll(mask, 1, axis=1)

    for i in range(depth):
        views = (
                    new_view[t_mask],
                    new_view[b_mask],
                    new_view[l_mask],
                    new_view[r_mask],
                    new_view[mask]
                )
        new_view[mask] += explTransform(views, alpha)

        new_view[~mask] = view[~mask]

        # Neumann boundary condition du/dt = 0
        new_view[0, :] = new_view[1, :]
        new_view[:, 0] = new_view[:, 1]
        new_view[-1, :] = new_view[-2, :]
        new_view[:, -1] = new_view[:, -2]

    return (new_img * 255).astype(np.uint8)


def implisitt(img, depth, mask, alpha):
    """
    Runs an implisitt diffusion of an image

    Paramters
    ---------
    img : np.ndarray
        Image to be diffused

    depth : int
        Number of itterations to run the diffusion

    mask : np.ndarray
        Boolean array for where to run diffusion.

    alpha : float
        delta_t / delta_x**2

    Returns
    -------
    np.ndarray
        Diffused image
    """
    new_img = img.astype(float) / 255
    img = img.astype(float) / 255
    mask = mask.astype(bool)
    maskCords = np.argwhere(mask)
    size = img.shape[:2]

    # Reduce image size to region diffuse region, performance improvement
    top = np.amin(maskCords[:, 0])
    bottom = np.amax(maskCords[:, 0]) + 1
    left = np.amin(maskCords[:, 1])
    right = np.amax(maskCords[:, 1]) + 1
    view = img[top:bottom, left:right]
    new_view = new_img[top:bottom, left:right]
    mask = mask[top:bottom, left:right]

    shape = view.shape
    # Create 1D array for matrix opperation on whole array
    view = view.ravel()
    new_view = new_view.ravel()
    mask = mask.ravel()

    size = view.shape[0]
    nupperdiag = np.concatenate((np.zeros(shape[1]),
                                -alpha * np.ones(size - shape[1])))
    upperdiag = np.concatenate(([0, 0], -alpha * np.ones(size - 2)))
    centerdiag = np.concatenate((
                                    [1],
                                    (1 + 4 * alpha) * np.ones(size - 2),
                                    [1]
                                ))
    lowerdiag = np.concatenate((-alpha * np.ones(size - 2), [0, 0]))
    nlowerdiag = np.concatenate((-alpha * np.ones(size - shape[1]),
                                np.zeros(shape[1])))
    diag = np.array([nupperdiag, upperdiag, centerdiag, lowerdiag, nlowerdiag])
    sparse = spdiags(diag, [shape[1], 1, 0, -1, -shape[1]], size, size).tocsc()

    for i in range(depth):
        new_view = spsolve(sparse, new_view)
        new_view[~mask] = view[~mask]

    new_view = np.asarray(new_view).reshape(shape)
    new_img[top:bottom, left:right] = new_view

    return (new_img * 255).astype(np.uint8)


if __name__ == "__main__":
    img = np.array(imageio.imread("../../Small_grayScale.png"))
    #img = np.array(imageio.imread("test.png"))
    img[1:4, 1:9] = 0
    n = 2
    mask = np.ones((5, 10)).astype(bool)
    mask[2:4, 3:6] = False
    alpha = 0.5
    #print(img)
    img[:, :, 0] = implisitt(img[:, :, 0], n, mask, alpha)
    #img[:, :, 1] = implisitt(img[:, :, 1], n, mask, alpha)
    #img[:, :, 2] = implisitt(img[:, :, 2], n, mask, alpha)
    #print(new_img)
