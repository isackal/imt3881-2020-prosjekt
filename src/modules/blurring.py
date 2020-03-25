import modifiers as md
import numpy as np


def blurring(img, n, alpha, mask):
    """
    Blurs the image

    Alpha should remain below 0.25 to prevent numeric
    instablilty.

    Paramters
    ---------
    img : np.ndarray
        Source image
    n : int
        Number of iterations (default = 10)
    alpha : float
        delta_t / delta_x**2 (default = 0.25)

    Returns
    -------
    np.ndarray
        Blurred image
    """

    new_img = img.astype(float) / 255
    centerMask = mask.astype(bool)

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
        laplace = (new_img[topMask] +
                   new_img[botMask] +
                   new_img[leftMask] +
                   new_img[rightMask] -
                   4 * new_img[centerMask])

        new_img[centerMask] += alpha * laplace

        # Neumann boundary condition du/dt = 0
        new_img[0, :] = new_img[1, :]
        new_img[-1, :] = new_img[-2, :]
        new_img[:, 0] = new_img[:, 1]
        new_img[:, -1] = new_img[:, -2]

    return (new_img * 255).astype(np.uint8)


class Blurring(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Blurring"
        self.function = blurring
        self.params = [
            ("img", np.ndarray, None),
            ("iterations", int, 10),
            ("alpha", float, 0.25),
            ("mask", np.ndarray, None)
        ]
        self.initDefaultValues()
