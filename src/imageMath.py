import modifiers as md
import numpy as np
import formating as frm
import errorhandling as eh
import imageio as im
import diffusion as df


"""
The following modifiers play an essential role in the interactivity in the
app, and will therefore be included in the final product.
Eg. the noise module is meant for the user to be able to add
noise to images to test different smoothing methods.
"""


def rgbaFormat(o_image):
    if o_image.shape[2] == 4:
        return o_image.astype(np.uint8)
    else:
        _image = np.ones((o_image.shape[0], o_image.shape[1], 4)) * 255
        for i in range(3):
            _image[:, :, i] = o_image[:, :, i]  # RED
        return _image.astype(np.uint8)


def read(imgFile):
    return im.imread(imgFile).astype(np.float16)/255


def addNoise(img, seed, minVal, maxVal):
    np.random.seed(seed)
    _rng = maxVal - minVal  # Range of the noise
    _rand = np.random.rand(*list(img.shape))*_rng + minVal
    img2 = img*1
    img2[:, :, :3] += _rand[:, :, :3]
    return img2


def findEdges(img):
    return np.sqrt(df.gX(img)**2 + df.gY(img)**2)/np.sqrt(2)


def mosaic_get_green(img):
    # eh.showImageData(img, "input mosaic_get_green")
    x, y = np.mgrid[0:img.shape[0]:1, 0:img.shape[1]:1]
    z = ((x + y).astype(int) % 2) == 0
    # eh.showImageData(img[:, :, 1]*z, "output mosaic_get_green")
    return img[:, :, 1]*z


def mosaic_get_red(img):
    x, y = np.mgrid[0:img.shape[0]:1, 0:img.shape[1]:1]
    z = ((x.astype(int)+1)*y.astype(int)) % 2 == 1
    return img[:, :, 0]*z


def mosaic_get_blue(img):
    x, y = np.mgrid[0:img.shape[0]:1, 0:img.shape[1]:1]
    z = (x.astype(int)*(y.astype(int)+1)) % 2 == 1
    return img[:, :, 2]*z


def mosaic_get(img, r, g, b):
    # eh.showImageData(img, "input mosaic_get")
    img2 = np.ones(img.shape)
    img2[:, :, 0] = mosaic_get_red(img)*r
    img2[:, :, 1] = mosaic_get_green(img)*g
    img2[:, :, 2] = mosaic_get_blue(img)*b
    # eh.showImageData(img2, "output mosaic_get")
    return img2


def mask_rand(u):
    """
    Finds the outer edges of a mask and returns it.
    """
    u1 = np.copy(u)
    u1[0:-1, :] += u[1:, :]
    u1[1:, :] += u[0:-1, :]
    u1[:, 0:-1] += u[:, 1:]
    u1[:, 1:] += u[:, 0:-1]
    return u1 ^ u


def diff(img1, img2):
    i1, i2 = frm.makeSameSize(img1, img2)
    i1[:, :, :3] -= i2[:, :, :3]
    i1[:, :, :3] = np.sqrt(i1[:, :, :3]**2)
    return i1


def normalize(img):
    _max = img[:, :, :3].max()
    k = 1./_max
    newimg = img*1
    newimg[:, :, :3] = img[:, :, :3]*k
    return newimg


def crop(img, x, y, width, height):
    while x < 0:
        x += img.shape[1]
    while y < 0:
        y += img.shape[0]
    if x + width >= img.shape[1]:
        width = img.shape[1] - 1 - x
    if y + height >= img.shape[0]:
        height = img.shape[0] - 1 - y
    return img[y:y+height, x:x+width, :]


def weightedAddition(img1, img2, weight, ignoreTransparency):
    i1, i2 = frm.makeSameSize(img1, img2)
    if ignoreTransparency != 0:
        i1[:, :, :3] += i2[:, :, :3] * weight
        return i1
    else:
        return i1+i2*weight


def multiplication(img1, img2):
    i1, i2 = frm.makeSameSize(img1, img2)
    return i1*i2


def exponent(img1, pwr):
    return img1**pwr


def grayWeighted(
    img,
    weightRed,
    weightGreen,
    weightBlue,
    weightAlpha
):
    weightSize = (
        weightRed +
        weightGreen +
        weightBlue +
        weightAlpha
    )
    grayscale = None
    if weightSize < 0.1:
        grayscale = (np.sum(img[:, :, :3], axis=2)/3)
    else:
        grayscale = (
            img[:, :, 0] * weightRed +
            img[:, :, 1] * weightGreen +
            img[:, :, 2] * weightBlue +
            img[:, :, 3] * weightAlpha
        ) / weightSize
    return grayscale


def grayVector(
    img
):
    grayscale = np.sqrt(
        np.sum(img[:, :, :3]**2, axis=2)
        )/np.sqrt(3)
    return grayscale


def grayToRGBA(img):
    newImg = np.zeros((
        img.shape[0],
        img.shape[1],
        4
        ), float)
    for i in range(3):
        newImg[:, :, i] += img
    newImg[:, :, 3] += 255
    return newImg


def colorFilter(img, r, g, b, ar, ag, ab):
    newImg = img*1
    _r = 1. * r / 100
    _g = 1. * g / 100
    _b = 1. * b / 100
    newImg[:, :, 0] = newImg[:, :, 0] * _r + ar
    newImg[:, :, 1] = newImg[:, :, 1] * _g + ag
    newImg[:, :, 2] = newImg[:, :, 2] * _b + ab
    return newImg


def makeMask(
    img,
    weightRed,
    weightGreen,
    weightBlue,
    weightAlpha,
    threshold
):
    gsc = grayWeighted(
            img,
            weightRed,
            weightGreen,
            weightBlue,
            weightAlpha
        ) > threshold
    return gsc


def invert(img):
    ret = img*1
    ret[:, :, :3] = -ret[:, :, :3]+1
    return ret


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
# TODO implement image moving


class WeightedAddition(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Weight Addition"
        self.function = weightedAddition
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("image", np.ndarray, None, md.FORMAT_RGBA),
            ("weight", float, 1.),
            ("ignore alpha", int, 1),
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class FitSize(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Fit Size"
        self.function = fitSizeOfImage
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("image", np.ndarray, None, md.FORMAT_RGBA)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class Offset(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Offset"
        self.function = offset
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("y", int, 0),
            ("x", int, 0)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class ColorToGrayWeighted(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "ColorToGrayWeighted"
        self.function = grayWeighted
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("Red", float, 1),
            ("Green", float, 1),
            ("Blue", float, 1),
            ("Alpha", float, 0)
        ]
        self.outputFormat = md.FORMAT_BW
        self.initDefaultValues()


class Binary(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Binary"
        self.function = makeMask
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("Red", float, 1),
            ("Green", float, 1),
            ("Blue", float, 1),
            ("Alpha", float, 0),
            ("Threshold", float, 0.5)
        ]
        self.outputFormat = md.FORMAT_BOOL
        self.initDefaultValues()


class Invert(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Invert"
        self.function = invert
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class VecotrGray(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Vector Gray"
        self.function = grayVector
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),

        ]
        self.outputFormat = md.FORMAT_BW
        self.initDefaultValues()


class Multiplication(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Multiplication"
        self.function = multiplication
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("image", np.ndarray, None, md.FORMAT_RGBA)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class Exponent(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Exponent"
        self.function = exponent
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("Power", float, 2)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class Diff(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Diff"
        self.function = diff
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("Image", np.ndarray, None, md.FORMAT_RGBA)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class Normalize(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Normalize"
        self.function = normalize
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class ColorFilter(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Color Filter"
        self.function = colorFilter
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("Red percent", float, 100),
            ("Green percent", float, 100),
            ("Blue percent", float, 100),
            ("Add Red", float, 0),
            ("Add Green", float, 0),
            ("Add Blue", float, 0)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class Mosaic(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Color Filter"
        self.function = mosaic_get
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("Red", float, 1.),
            ("Green", float, 1.),
            ("Blue", float, 1.)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


class Noise(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Noise"
        self.function = addNoise
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("seed", int, 0),
            ("min", float, -0.1),
            ("max", float, 0.1)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()

class FindEdges(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Find Edges"
        self.function = findEdges
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGB)
        ]
        self.outputFormat = md.FORMAT_RGB
        self.initDefaultValues()

class Crop(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Crop"
        self.function = crop
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("x", int, 0),
            ("y", int, 0),
            ("width", int, 32),
            ("height", int, 32)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()
