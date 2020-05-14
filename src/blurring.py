import numpy as np

import modifiers as md
import diffusion


def blurring(img, itr, alpha, mask, met=0):
    """
    Blurs the image

    Alpha should remain below 0.24 to prevent numeric
    instablilty for method = 0 (explisitt), remain below 0.48.
    for method = 1 (crank nicolson)
    The function is generalized in diffusion.py
    blurring.py is therefore only a call on to the actual diffusion process.

    Paramters
    ---------
    img : numpy.ndarray
        Source image
    n : int
        Number of iterations (default = 10)
    alpha : float
        delta_t / delta_x**2 (default = 0.24)
    met   : <int>
            default =>  explisitt method
            1       =>  crank nicolson
            2       =>  implisitt
            3       =>  direct solution


    Returns
    -------
    numpy.ndarray
        Blurred image
    """
    if met == 1:
        """
        we convert iterations and alpha to a more efficient
        equivelent for crank nicolson
        """
        _itr = (itr // 2) + 1
        _img = np.copy(img)

        alpha = alpha*(1.*itr/_itr)
        while alpha > 0.4999:
            alpha /= 2
            _itr *= 2
        for i in range(_itr):
            _img = np.clip(diffusion.poissonCrank(_img, 0, alpha), 0, 1)
        return _img
    elif met == 2:
        """
        we convert iterations and alpha to a more efficient
        equivelent for implisitt method
        """
        alpha *= itr
        return diffusion.poissonImplisitt(img, 0, alpha)
    elif met == 3:
        alpha *= itr
        lmb = 1. / (1 + alpha)
        return diffusion.directSolve(img, lmb)
    else:
        if mask is None:  # Blur whole image if no mask is given
            mask = np.ones(img.shape[:2])

        return diffusion.pre_diffuse(img, mask, alpha=alpha, itr=itr, met='e')


class Blurring(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Blurring"
        self.function = blurring
        self.params = [
            ("img", np.ndarray, None, md.FORMAT_RGB),
            ("iterations", int, 10),
            ("alpha", float, 0.24),
            ("mask", np.ndarray, None, md.FORMAT_BOOL),
            ("method: 0ex 1CN 2im 3dr", int, 1)
        ]
        self.outputFormat = md.FORMAT_RGB
        self.initDefaultValues()
        # Setting flags must be called after defualt values has been set:
        self.setFlags(3, md.FLAG_CAN_BE_NULL)
