import modifiers as md
import numpy as np
import formating as frm

"""
The following modifiers play an essential role in the interactivity in the
app, and will therefore be included in the final product.
Eg. the procedural module is meant for the user to be able to add
noise to images to test different smoothing methods.
"""

# The following functions are used to create noise ---------


def iSqrt(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def modulo(B, p, m):
    N = 3
    q = iSqrt(p)
    if type(p) is int:
        if p > N:
            return ((modulo(modulo(B % m, q, m), q, m) * modulo(B % m, p - q * q, m)) % m)
        else:
            return ((B**p) % m)
    else:
        _mask = p > N
        _mask2 = not _mask
        ret = np.zeros(p.shape)
        ret[_mask] = ((modulo(modulo(B % m, q, m), q, m) * modulo(B % m, p - q * q, m)) % m)
        ret[_mask2] = ((B**p) % m)
        return ret


def procedural(img, numb, pwr):
    tmp = modulo(img[:, :, :3].astype(int)*numb, pwr, 256)
    ret = np.ones(img.shape, int)*255
    ret[:, :, :3] = tmp
    return ret

# -----------------------------------------------------------------


def diff(img1, img2):
    i1, i2 = frm.makeSameSize(img1, img2)
    i1 = i1.astype(float)
    i1[:, :, :3] -= i2[:, :, :3].astype(float)
    i1[:, :, :3] = np.sqrt(i1[:, :, :3]**2)
    return i1.astype(np.uint8)


def normalize(img):
    _max = img[:, :, :3].max()
    k = 255./_max
    newimg = img.astype(int)
    newimg[:, :, :3] = img[:, :, :3]*k
    return newimg.astype(np.uint8)


def weightedAddition(img1, img2, weight, ignoreTransparency):
    i1, i2 = frm.makeSameSize(img1, img2)
    if ignoreTransparency != 0:
        i1[:, :, :3] += i2[:, :, :3] * weight
        return i1
    else:
        return i1+i2*weight


def multiplication(img1, img2):
    i1, i2 = frm.makeSameSize(img1, img2)
    return (((i1.astype(float)/255) * (i2.astype(float)/255))*255).astype(np.uint8)


def exponent(img1, pwr):
    return (((img1.astype(float)/255)**pwr)*255).astype(np.uint8)


def grayWeighted(
    img,
    weightRed,
    weightGreen,
    weightBlue,
    weightAlpha
):
    img2 = img.astype(float)
    weightSize = (
        weightRed +
        weightGreen +
        weightBlue +
        weightAlpha
    )
    grayscale = None
    if weightSize < 0.1:
        grayscale = (np.sum(img2[:, :, :3], axis=2)/3).astype(np.uint8)
    else:
        grayscale = ((
            img2[:, :, 0] * weightRed +
            img2[:, :, 1] * weightGreen +
            img2[:, :, 2] * weightBlue +
            img2[:, :, 3] * weightAlpha
        ) / weightSize).astype(np.uint8)
    return grayscale


def grayVector(
    img
):
    img2 = img.astype(float)
    grayscale = (np.sqrt(
        np.sum(img2[:, :, :3]**2, axis=2)
        )/np.sqrt(3)).astype(np.uint8)
    return grayscale


def grayToRGBA(img):
    newImg = np.zeros((
        img.shape[0],
        img.shape[1],
        4
        ), int)
    for i in range(3):
        newImg[:, :, i] += img
    newImg[:, :, 3] += 255
    return newImg.astype(np.uint8)


def colorFilter(img, r, g, b):
    newImg = img.astype(float)
    _r = 1. * r / 100
    _g = 1. * g / 100
    _b = 1. * b / 100
    newImg[:, :, 0] *= _r
    newImg[:, :, 1] *= _g
    newImg[:, :, 2] *= _b
    return np.clip(newImg, 0, 255).astype(np.uint8)


def colorToGrayWeighted(
    img,
    weightedRed,
    weightedGreen,
    weightedBlue,
    weightedAlpha
):
    return grayToRGBA(
        grayWeighted(
            img,
            weightedRed,
            weightedGreen,
            weightedBlue,
            weightedAlpha
        )
    )


def colorToGrayVector(
    img
):
    return grayToRGBA(
        grayVector(
            img
        )
    )


def makeMask(
    img,
    weightRed,
    weightGreen,
    weightBlue,
    weightAlpha,
    threshold
):
    gsc = (
        grayWeighted(
            img,
            weightRed,
            weightGreen,
            weightBlue,
            weightAlpha
        ) > threshold
    ).astype(np.uint8)*255
    return grayToRGBA(gsc)


def invert(img):
    newImg = img*1
    newImg[:, :, :3] = 1 - newImg[:, :, :3]
    return newImg


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


class ColorToGrayWeighted(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "ColorToGrayWeighted"
        self.function = colorToGrayWeighted
        self.params = [
            ("source", np.ndarray, None),
            ("Red", float, 1),
            ("Green", float, 1),
            ("Blue", float, 1),
            ("Alpha", float, 0)
        ]
        self.initDefaultValues()


class Binary(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Binary"
        self.function = makeMask
        self.params = [
            ("source", np.ndarray, None),
            ("Red", float, 1),
            ("Green", float, 1),
            ("Blue", float, 1),
            ("Alpha", float, 0),
            ("Threshold", int, 127)
        ]
        self.initDefaultValues()


class Invert(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Invert"
        self.function = invert
        self.params = [
            ("source", np.ndarray, None)
        ]
        self.initDefaultValues()


class VecotrGray(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Vector Gray"
        self.function = colorToGrayVector
        self.params = [
            ("source", np.ndarray, None)
        ]
        self.initDefaultValues()


class Procedural(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Procedural"
        self.function = procedural
        self.params = [
            ("source", np.ndarray, None),
            ("Number", int, 1),
            ("Power", int, 327)
        ]
        self.initDefaultValues()


class Multiplication(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Multiplication"
        self.function = multiplication
        self.params = [
            ("source", np.ndarray, None),
            ("image", np.ndarray, None)
        ]
        self.initDefaultValues()


class Exponent(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Exponent"
        self.function = exponent
        self.params = [
            ("source", np.ndarray, None),
            ("Power", float, 2)
        ]
        self.initDefaultValues()


class Diff(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Diff"
        self.function = diff
        self.params = [
            ("source", np.ndarray, None),
            ("Image", np.ndarray, None)
        ]
        self.initDefaultValues()


class Normalize(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Normalize"
        self.function = normalize
        self.params = [
            ("source", np.ndarray, None)
        ]
        self.initDefaultValues()


class ColorFilter(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Color Filter"
        self.function = colorFilter
        self.params = [
            ("source", np.ndarray, None),
            ("Red percent", float, 100),
            ("Green percent", float, 100),
            ("Blue percent", float, 100)
        ]
        self.initDefaultValues()
