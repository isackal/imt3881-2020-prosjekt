import modifiers as md
import numpy as np
import formating as frm

"""
The following modifiers play an essential role in the interactivity in the
app, and will therefore be included in the final product.
"""


def weightedAddition(img1, img2, weight, ignoreTransparency):
    i1, i2 = frm.makeSameSize(img1, img2)
    if ignoreTransparency != 0:
        i1[:, :, :3] += i2[:, :, :3] * weight
        return i1
    else:
        return i1+i2*weight


def fitSizeOfImage(img1, img2):
    i1, i2 = frm.makeSameSize(img1, img2)
    return i1


def offset(img, ox, oy):
    _x = 8
    _y = 8
    while ox < 0:
        ox += img.shape[0] * _x
        _x *= 8
    while oy < 0:
        oy += img.shape[1] * _y
        _y *= 8
    ox = ox % img.shape[0]
    oy = oy % img.shape[1]
    ret = np.zeros(img.shape)
    w = img.shape[0] - ox
    h = img.shape[1] - oy
    ret[ox:, oy:, :] = img[:w, 0:h, :]
    ret[0:ox, 0:oy, :] = img[w:, h:, :]
    ret[0:ox, oy:] = img[w:, 0:h, :]
    ret[ox:, 0:oy] = img[0:w, h:, :]
    return ret

# TODO implement padding
# TODO implement scaling
# TODO implement cropping


class WeightedAddition(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Weight Addition"
        self.function = weightedAddition
        self.params = [
            ("source", np.ndarray, None),
            ("image", np.ndarray, None),
            ("weight", float, 1.),
            ("ignore alpha", int, 1)
        ]
        self.initDefaultValues()


class FitSize(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Fit Size"
        self.function = fitSizeOfImage
        self.params = [
            ("source", np.ndarray, None),
            ("image", np.ndarray, None)
        ]
        self.initDefaultValues()


class Offset(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Offset"
        self.function = offset
        self.params = [
            ("source", np.ndarray, None),
            ("y", int, 0),
            ("x", int, 0)
        ]
        self.initDefaultValues()
