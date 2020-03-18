import modifiers as md
import numpy as np


def blurring(img, n, alpha):
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

    for i in range(n):
        laplace = (new_img[2:, 1:-1] +
                   new_img[:-2, 1:-1] +
                   new_img[1:-1, 2:] +
                   new_img[1:-1, :-2] -
                   4 * new_img[1:-1, 1:-1])

        new_img[1:-1, 1:-1] += alpha * laplace

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
            ("alpha", float, 0.25)
        ]
        self.initDefaultValues()
