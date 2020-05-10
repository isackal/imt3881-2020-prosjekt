import diffusion as df
import numpy as np
import modifiers as md
import errorhandling as eh


def BWKantBevGlatting(u, alpha=0.24, k=0.1, itr=1):
    u1 = u*1  # Copy u so u is not modified:
    D = 1. / (1 + k * (df.gX(u1)**2 + df.gY(u1)**2))
    u1 = df.pre_diffuse(u1, met='e', rand='n', alpha=alpha, itr=itr, D=D)
    #for i in range(itr):
    #    u1 = df.diffuse(u1, alpha, 0, D)
    return u1


def RGBAKantBevGlatting(u, alpha=0.24, k=0.1, itr=1):
    timg = np.zeros(u.shape)  # Transformed image
    for i in range(3):
        timg[:, :, i] = BWKantBevGlatting(u[:, :, i], alpha, k, itr)
    if(u.shape[2] == 4):
        timg[:, :, 3] = BWKantBevGlatting(u[:, :, 3], alpha, k, itr)
    return timg


class KantbevarendeGlatting(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Kantbevarende Glatting"
        self.function = RGBAKantBevGlatting
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("alpha", float, 0.24),
            ("kantbevaring", float, 100),
            ("iterasjoner", int, 10)
        ]
        self.outputFormat = md.FORMAT_RGBA  # RGBA formatted output
        self.initDefaultValues()


if __name__ == "__main__":
    import imageio as im
    import matplotlib.pyplot as plt
    orig_im = im.imread("../testimages/raccoon.png") / 255
    tImg = RGBAKantBevGlatting(orig_im, 0.24, 11000, 100)
    plt.imshow(tImg)
    plt.show()
