import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import inv
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

    """
    shape = view.shape
    # Create 1D array for matrix opperation on whole array
    view = view.ravel()
    new_view = new_view.ravel()
    mask = mask.ravel()
    mask = np.append(mask, [mask, mask])

    size = view.shape[0]
    upperdiag = np.concatenate(([0, 0], -alpha * np.ones(size - 2)))
    centerdiag = np.concatenate((
                                    [1],
                                    (1 + 2 * alpha) * np.ones(size - 2),
                                    [1]
                                ))
    lowerdiag = np.concatenate((-alpha * np.ones(size - 2), [0, 0]))
    diag = np.array([upperdiag, centerdiag, lowerdiag])

    sparse = spdiags(diag, [1, 0, -1], size, size).tocsc()
    #invSparse = inv(sparse)
    for i in range(depth):
        res = bicgstab(sparse, new_view, new_view)
        if(res[1] == 0):
            new_view = res[0]
            new_view[~mask] = view[~mask]
        else:
            print(
                "The itterative solution did not work out.\
                \nTry an explisite solution to this picture"
                )
            return (img * 255).astype(np.uint8)

    new_view = np.asarray(new_view).reshape(shape)
    new_img[top:bottom, left:right] = new_view

    return (new_img * 255).astype(np.uint8)
    """

    # Create a sparse matrix for columns and rows in picture
    size = view.shape[:2]
    sparse = []
    for i in range(2):
        u_diag = np.concatenate(([0, 0], -alpha * np.ones(size[i] - 2)))
        c_diag = np.concatenate((
                                [1],
                                (1 + 2 * alpha) * np.ones(size[i] - 2),
                                [1]
                                ))
        l_diag = np.concatenate((-alpha * np.ones(size[i] - 2), [0, 0]))
        diag = np.array([u_diag, c_diag, l_diag])
        sparse.append(spdiags(diag, [1, 0, -1], size[i], size[i]).tocsc())

    # Diffuse image
    for i in range(depth):
        # Diffuse each column sequencially
        # Possible to do all cols at the same time?
        for j in range(size[1]):
            new_view[:, j] = spsolve(sparse[0], new_view[:, j])

        # Diffuse each row sequencially
        # Possible to do all rows or everything at once?
        for k in range(size[0]):
            new_view[k, :] = spsolve(sparse[1], new_view[k, :])

        # Return values of image to original where needed
        new_view[~mask] = view[~mask]

    # Return diffused image
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
    new_img = implisitt(img, n, mask, alpha)
    #print(new_img)
