import numpy as np
import modifiers as md


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
