import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve
from linalgSolvers import solveAsnRgb


def explisittPart(u, h, a):
    """
    The right side of a crank nicolson equation, or can be
    used to diffuse something.

    Parameters
    ----------

    u       :   <ndarray>
                image

    h       :   <float>, <ndarray>
                h part in the poisson equation

    alpha   :   <float>
                alpha = dt/dx^2, alpha part of the poisson equation

    Returns
    -------

    <ndarray>
    image result
    """
    # Kan senere erstattes med Aslaks diffuse
    u1 = -4*u
    u1[0:-1, :] += u[1:, :]
    u1[1:, :] += u[0:-1, :]
    u1[:, 0:-1] += u[:, 1:]
    u1[:, 1:] += u[:, 0:-1]
    # Neumann condition
    u1[0, :] += u[0, :]
    u1[:, 0] += u[:, 0]
    u1[-1, :] += u[-1, :]
    u1[:, -1] += u[:, -1]
    # return (u1 * a + h * a + u)
    return u + (u1*a) - (h*a)


def directSolve(u, l, dx=1):
    return solveAsnRgb(
        4+l*dx*dx,
        -1,
        u*l*dx*dx,
        solver="b"
    )


def poissonImplisitt(
    u,
    h=0,
    alpha=0.2  # Can be very high
):
    """
    Iterative solution of the poisson equation with Implisitt solution.
    Does one iteration.

    Parameters
    ----------

    u       :   <ndarray>
                image

    h       :   <float>, <ndarray>
                h part in the poisson equation

    alpha   :   <float>
                alpha = dt/dx^2, alpha part of the poisson equation

    Returns
    -------

    <ndarray>
    image result
    """
    return solveAsnRgb(
        4*alpha + 1,
        -alpha,
        u - h*alpha
    )


def poissonCrank(
    u,
    h=0,
    alpha=0.4  # Best if alpha < 0.5
):
    """
    Iterative solution of the poisson equation with Crank Nicolson.
    Does one iteration.

    Parameters
    ----------

    u       :   <ndarray>
                image

    h       :   <float>, <ndarray>
                h part in the poisson equation

    alpha   :   <float>
                alpha = dt/dx^2, alpha part of the poisson equation
                should be less than 0,5. 0,5 may be slightly unstable.

    Returns
    -------

    <ndarray>
    image result
    """
    return solveAsnRgb(
        2 * alpha + 1,
        -alpha / 2,
        explisittPart(u, h, alpha/2)
    )


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


def pre_diffuse(u, mask=None, met='e', alpha=0.24, itr=50, h=0, D=1):
    """
    Preps an image for diffusion

    Generates a view of the image around the truth values for the
    boolean mask this speeds up processing of images if the boolean mask
    is not true everywhere.

    Parameters
    ----------
    u : <numpy.ndarray>
        image to be processed
    mask : <numpy.ndarray>
        boolean mask of where to perform diffusion
    met : char
        the schema used to diffuse, (e)xplicit or (i)mplicit

    alpha : float
        delta_t / delta_x**2 (default = 0.24)

    itr : int
        number of times the diffusion should run (default  = 50)

    h : <numpy.ndarray>
        extra h value that should be added to each pixel in u
        is not added if h is not a numpy.ndarray (default = 0)

    d : <numpy.ndarray>
        Scalefactor for edges, used to scale down diffusion around edges if
        requested. Ignored if not of type numpy.ndarray (default = 0)
    Returns
    -------
    u1 : <numpy.ndarray>
        diffused image
    """
    # u1 for returned img, u for returning pixels to origian value if wanted
    u1 = np.copy(u)

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

            new_view[new_view > 1] = 1
            new_view[new_view < 0] = 0

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

    u1[top:bottom, left:right] = new_view

    return u1


def explicit(u, mask, laplace=0, alpha=0.24, h=0, D=1):
    """
    runs one iteration trough an explicit schema

    Alpha should remain below 0.24 to prevent numeric
    instablilty.

    Paramters
    ---------
    u : np.ndarray
        View of source image
    mask : numpy.ndarray
        boolean mask where values should and should not be added
    laplace : numpy.ndarray
        laplace transformation to be applied to view
    alpha : float
        delta_t / delta_x**2 (default = 0.24)
    h : int / numpy.ndarray
        array of diffrent h values to subtract from view
        ignored if not numpy.ndarray
    D : int / numpy.ndarray
        array of scale factors to apply to laplace transformation
        ignored if not numpy.ndarray

    Returns
    -------
    np.ndarray
        image with one laplace transformation
    """
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
    """
    Solves poisson schema implicitly

    Uses sparse matricies to solve Ax = b. Currently not in use
    as boundry condition is not properly handled

    Paramters
    ---------
    sparse : sparse matrix
        Sparse matrix for A
    u : numpy.ndarray
        Vectorized version of image, used as b
    shape : numpy.ndarray
        shape of original image, used for skipping to next row/col of image

    Returns
    -------
    np.ndarray
        image with one laplace transformation
    """
    u = spsolve(sparse, u)
    u[0::shape[1]] = u[1::shape[1]]
    u[shape[1]-1::shape[1]] = u[shape[1]-2::shape[1]]
    return u


if __name__ == "__main__":
    import imageMath as im
    import matplotlib.pyplot as plt
    orig_im = im.read("../testimages/raccoon.jpg")
    print("Running Crank Poisson")
    img2 = np.copy(orig_im)
    for i in range(10):
        print(i+1)
        img2 = poissonCrank(img2, alpha=0.49)
    print("Running Direkte glatting")
    img1 = directSolve(orig_im, 0.001)
    print("Running Implisitt Poisson")
    img3 = poissonImplisitt(orig_im, alpha=1000)
    plt.imshow(img2)
    plt.show()
    plt.imshow(img1)
    plt.show()
    plt.imshow(img3)
    plt.show()
