import modifiers as md
import numpy as np
import poisson


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

    return poisson.explisitt(img, n, mask, alpha)


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
