import modifiers as md
import numpy as np


def blurring(img, n, alpha, mask):
    """
    Blurs the image

    Alpha should remain below 0.24 to prevent numeric
    instablilty.

    Paramters
    ---------
    img : np.ndarray
        Source image
    n : int
        Number of iterations (default = 10)
    alpha : float
        delta_t / delta_x**2 (default = 0.24)

    Returns
    -------
    np.ndarray
        Blurred image
    """
    if mask is None:  # Blur whole image if no mask is given
        mask = np.ones(img.shape[:2])

    new_img = img.astype(float) / 255
    img = img.astype(float) / 255
    mask = mask.astype(bool)
    centerMask = mask

    # Ensure blurring is not attempted directly on the boundary
    centerMask[0, :] = False
    centerMask[:, 0] = False
    centerMask[-1:, :] = False
    centerMask[:, -1:] = False

    # Create diffrent views of blurring region for laplace
    topMask = np.roll(centerMask, -1, axis=0)
    botMask = np.roll(centerMask, 1, axis=0)
    leftMask = np.roll(centerMask, -1, axis=0)
    rightMask = np.roll(centerMask, 1, axis=1)

    for i in range(n):
        laplace = (new_img[2:, 1:-1] +
                   new_img[:-2, 1:-1] +
                   new_img[1:-1, 2:] +
                   new_img[1:-1, :-2] -
                   4 * new_img[1:-1, 1:-1])
        new_img[1:-1, 1:-1] =0# alpha * laplace
        # Neumann boundary condition du/dt = 0
        new_img[0, :] = new_img[1, :]
        new_img[-1, :] = new_img[-2, :]
        new_img[:, 0] = new_img[:, 1]
        new_img[:, -1] = new_img[:, -2]

        # revert sections of image not ment to be blurred
        new_img[~mask] = img[~mask]

    return (new_img * 255).astype(np.uint8)


class Blurring(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Blurring"
        self.function = blurring
        self.params = [
            ("img", np.ndarray, None),
            ("iterations", int, 10),
            ("alpha", float, 0.24),
            ("mask", np.ndarray, None)
        ]
        self.initDefaultValues()
