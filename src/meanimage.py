import numpy as np
import modifiers as md

# This is another test module that can be removed later in the project.


def meanimage(img1, img2):
    if img2 is not None:
        maxwidth = img1.shape[1]
        maxheight = img1.shape[0]
        if img2.shape[1] > maxwidth:
            maxwidth = img2.shape[1]
        if img2.shape[0] > maxheight:
            maxheight = img2.shape[0]
        newImg = np.zeros((maxheight, maxwidth, 4))
        newImg[:img1.shape[0], :img1.shape[1], :] = img1.astype(float)*0.5
        newImg[:img2.shape[0], :img2.shape[1], :] += img2.astype(float)*0.5
        return newImg.astype(np.uint8)
    else:
        return img1


class Meanimage(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Mean Image"
        self.function = meanimage
        self.params = [
            ("source", np.ndarray, None),
            ("image", np.ndarray, None)
        ]
        self.initDefaultValues()
