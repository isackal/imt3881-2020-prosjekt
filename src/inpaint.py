import numpy as np

import modifiers as md
import diffusion


def inpaint(img, itr, mask, alpha):
    """
    Blurs the image

    Alpha should remain below 0.25 to prevent numeric
    instablilty.

    Paramters
    ---------
    img : np.ndarray
        Source image
    depth : int
        Number of iterations (default = 50)
    mask : np.ndarray
        Region of image that should be inpainted
    Returns
    -------
    np.ndarray
        Inpainted image
    """
    if mask is None:  # No inpaint region specified, do no inpainting
        return img
    else:
        return diffusion.pre_diffuse(img, mask, alpha=alpha, itr=itr, met='e')


class Inpaint(md.Modifier):
    #   read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "Inpaint"
        self.function = inpaint
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("depth", int, 50),
            ("mask", np.ndarray, None, md.FORMAT_BOOL),
            ("alpha", float, 0.2)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()
