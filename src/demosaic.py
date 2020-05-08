import numpy as np
import modifiers as md

from inpaint import inpaint

#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def demosaic(red, green, blue):
    redMask = ~red.astype(bool)
    greenMask = ~green.astype(bool)
    blueMask = ~blue.astype(bool)
    img = np.zeros((red.shape[0], red.shape[1], 3))

    new_red = inpaint(red, 50, redMask, 0.24)
    new_green = inpaint(green, 50, greenMask, 0.24)
    new_blue = inpaint(blue, 50, blueMask, 0.24)

    img[:, :, 0] = new_red
    img[:, :, 1] = new_green
    img[:, :, 2] = new_blue

    return img.astype(np.uint8)


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


#   Testfunksjon. Slett ved endelig release
if __name__ == "__main__":
    img = imageio.imread('../../face.png')
    red = np.zeros(img.shape[:2])
    blue = np.zeros(img.shape[:2])
    green = np.zeros(img.shape[:2])
    red[::2, ::2] = img[::2, ::2, 0]
    green[1::2, ::2] = img[1::2, ::2, 1]
    green[::2, 1::2] = img[::2, 1::2, 1]
    blue[1::2, 1::2] = img[1::2, 1::2, 2]

    new_img = demosaic(red, green, blue)
    plt.imshow(new_img)
    plt.show()
