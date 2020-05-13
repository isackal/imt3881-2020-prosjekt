import numpy as np
import scipy.ndimage as ndi
import modifiers as md


def _map(x, arr):
    return np.take(arr, x)


def bitcrusher(img, val):
    img1 = (img*255).astype(int)
    return np.round(255*np.round((img1.astype(np.float) / 255) * val) / val).astype(float) / 255


class Bitcrusher(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Bitcrusher"
        self.function = bitcrusher
        # First parameter should always be source image by default.
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("new max", int, 1)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()

if __name__ == "__main__":
    x = np.arange(0, 256, 1, np.int32)
    g = x**2
    y = (np.random.rand(500)*256).astype(np.int32)
    xx = np.array([[1,2,1,1,2,3,2,3],[2,2,3,3,2,3,4,3]])
    _mask = (xx==2)
    _mask = _mask == False
    xx[_mask] += xx[_mask]*4
    z = _map(xx, g)
    print(z)