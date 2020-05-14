import modifiers as md
import numpy as np


def contrast(img, n=10, k=5, alpha=0.24):
    """
    Amlpifies the contrast in the picture

    Finds the same picture with an amplified gradient.
    Alpha should not be higher than 0.25 to prevent numeric
    instability. A high k value will also cause a lot of
    noise.

    Paramters
    ---------
    img : np.ndarray
        Source image
    n : int
        Number of iterations (default = 10)
    k : float
        Steepness of the new gradient (default = 5)
    alpha : float
        delta_t / delta_x**2 (default = 0.24)

    Returns
    -------
    np.ndarray
        Image with amplified contrast
    """
    new_img = img
    laplace_0 = (new_img[2:, 1:-1] +
                 new_img[:-2, 1:-1] +
                 new_img[1:-1, 2:] +
                 new_img[1:-1, :-2] -
                 4 * new_img[1:-1, 1:-1])

    for i in range(n):
        laplace = (new_img[2:, 1:-1] +
                   new_img[:-2, 1:-1] +
                   new_img[1:-1, 2:] +
                   new_img[1:-1, :-2] -
                   4 * new_img[1:-1, 1:-1])

        new_img[1:-1, 1:-1] += alpha * (laplace - k * laplace_0)

    # Trim values outside scope

    new_img[new_img > 1] = 1
    new_img[new_img < 0] = 0

    return new_img


class Contrast(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Contrast"
        self.function = contrast
        self.params = [
            ("img", np.ndarray, None, md.FORMAT_RGBA),
            ("iterations", int, 10),
            ("steepness", float, 5.0),
            ("alpha", float, 0.24)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()
