import numpy as np

import modifiers as md
from inpaint import inpaint


def demosaic(red, green, blue):
    """
    Inpaints 3 one channel images into one RGB image

    Method assumes a Bayer-like arrangement of pixels
        "https://en.wikipedia.org/wiki/Bayer_filter"

    Paramters
    ---------
    red : numpy.ndarray
        red channel
    green : numpy.ndarray
        green channel
    blue : numpy.ndarray
        blue channel

    Returns
    -------
    np.ndarray
        Demosaiced image
    """
    # Create boolean masks true where R, G and B do not have values
    redMask = ~red.astype(bool)
    greenMask = ~green.astype(bool)
    blueMask = ~blue.astype(bool)

    # Create return image
    img = np.zeros((red.shape[0], red.shape[1], 3))

    # Inpaint channels
    new_red = inpaint(red, 10, redMask, 0.24)
    new_green = inpaint(green, 10, greenMask, 0.24)
    new_blue = inpaint(blue, 10, blueMask, 0.24)

    # Insert inpainted values in new image
    img[:, :, 0] = new_red
    img[:, :, 1] = new_green
    img[:, :, 2] = new_blue

    return img


class Demosaic(md.Modifier):
    #   read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "demosaic"
        self.function = demosaic
        self.params = [
            ("red", np.ndarray, None, md.FORMAT_RED),
            ("green", np.ndarray, None, md.FORMAT_GREEN),
            ("blue", np.ndarray, None, md.FORMAT_BLUE)
        ]
        self.outputFormat = md.FORMAT_RGB
        self.initDefaultValues()
