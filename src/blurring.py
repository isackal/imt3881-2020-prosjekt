import numpy as np

import modifiers as md
import diffusion


def blurring(img, itr, mask,  alpha):
    """
    Blurs the image

    Alpha should remain below 0.24 to prevent numeric
    instablilty. The function is generalized in diffusion.py
    blurring.py is therefore only a call on to the actual diffusion process.

    Paramters
    ---------
    img : numpy.ndarray
        Source image
    n : int
        Number of iterations (default = 10)
    alpha : float
        delta_t / delta_x**2 (default = 0.24)

    Returns
    -------
    numpy.ndarray
        Blurred image
    """
    if mask is None:  # Blur whole image if no mask is given
        mask = np.ones(img.shape[:2])

    return diffusion.pre_diffuse(img, mask, alpha=alpha, itr=itr, met='e')


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
