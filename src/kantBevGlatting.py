import diffusion as df
import numpy as np
import modifiers as md
import matplotlib.pyplot as plt


def getD(u, k):
    D = 1. / (1 + k * (df.gX(u)**2 + df.gY(u)**2))
    return D


def BWKantBevGlatting(u, alpha=0.24, k=0.1, itr=1):
    u1 = u*1  # Copy u so u is not modified:
    D = getD(u1, k)
    u1 = df.pre_diffuse(u1, met='e', alpha=alpha, itr=itr, D=D)
    return u1


def RGBAKantBevGlatting(u, alpha=0.24, k=0.1, itr=1):
    timg = np.zeros(u.shape)  # Transformed image
    # Prevent bug if image is 1 channel only
    if(len(u.shape) == 3):
        for i in range(u.shape[2]):
            timg[:, :, i] = BWKantBevGlatting(u[:, :, i], alpha, k, itr)
    else:
        timg = BWKantBevGlatting(u, alpha, k, itr)
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


def testKBG():
    import imageio as im
    # import matplotlib.pyplot as plt
    orig_im = im.imread("../testimages/raccoon.png").astype(float) / 255
    tImg = RGBAKantBevGlatting(orig_im, 0.24, 11000, 100)
    plt.imshow(tImg)
    plt.show()


if __name__ == "__main__":
    testKBG()
