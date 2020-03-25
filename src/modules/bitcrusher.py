import numpy as np
import modifiers as md


def bitcrusher(img, val):
    return np.round(255*np.round((img.astype(np.float) / 255) * val) / val).astype(np.uint8)


class Bitcrusher(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Bitcrusher"
        self.function = bitcrusher
        # First parameter should always be source image by default.
        self.params = [
            ("source", np.ndarray, None),
            ("new max", int, 1)
        ]
        self.initDefaultValues()
