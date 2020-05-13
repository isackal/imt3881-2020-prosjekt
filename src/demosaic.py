import numpy as np

import modifiers as md
from inpaint import inpaint


def demosaic(red, green, blue):
    redMask = ~red.astype(bool)
    greenMask = ~green.astype(bool)
    blueMask = ~blue.astype(bool)
    img = np.zeros((red.shape[0], red.shape[1], 3))

    new_red = inpaint(red, 10, redMask, 0.24)
    new_green = inpaint(green, 10, greenMask, 0.24)
    new_blue = inpaint(blue, 10, blueMask, 0.24)

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
            ("red", np.ndarray, None),
            ("green", np.ndarray, None),
            ("blue", np.ndarray, None)
        ]
        self.initDefaultValues()
