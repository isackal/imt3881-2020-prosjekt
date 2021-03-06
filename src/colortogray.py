import numpy as np
import diffusion as df
import modifiers as md
import imageMath as imth
import errorhandling as eh
import imageio as im

import matplotlib.pyplot as plt


def color_to_gray(u, alpha=0.2, itr=3, _k=1., epsilon=0.0001):
    u0 = u*1
    length = np.zeros(u[:, :, 0].shape)  # Gradienten lengde
    vx = length * 0  # Retningen i x retning
    vy = length * 0  # Retningen i y retning
    colSum = length * 0
    _max = np.sqrt(3)  # sqrt(3) if 3 channels (RGB)

    for i in range(3):
        _gX = df.gX(u0[:, :, i])
        _gY = df.gY(u0[:, :, i])
        length += _gX**2 + _gY**2
        colSum += u0[:, :, i]
    length = np.sqrt(length) / _max
    vx = df.gX(colSum) + epsilon
    vy = df.gY(colSum)
    cSize = np.sqrt(vx**2 + vy**2)
    vx = length * vx / cSize
    vy = length * vy / cSize

    h = df.gX(vx) + df.gY(vy)
    u1 = np.sum(u0, axis=2) / 3  # Initial case

    u1 = df.pre_diffuse(u1, alpha=0.24, h=(h*_k), itr=itr)
    return np.clip(u1, 0, 1)


class ColorToGray(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Color To Gray"
        self.function = color_to_gray
        self.params = (
            ("source", np.ndarray, None, md.FORMAT_RGB),
            ("Alpha", float, 0.2),
            ("Iterations", int, 3),
            ("K", float, 1)
        )
        self.outputFormat = md.FORMAT_BW
        self.initDefaultValues()


if __name__ == "__main__":
    img = im.imread("../hdr-bilder/Ocean/Ocean_02048.png").astype(float)/255
    gray = color_to_gray(img, _k=5)
    plt.imshow(gray, plt.cm.gray)
    plt.show()
