import diffusion as df
import numpy as np
import modifiers as md


def BWKantBevGlatting(u, alpha=0.24, k=0.1, itr=1):
    u1 = u.astype(float)/255
    D = 1. / (1 + k * (df.gX(u1)**2 + df.gY(u1)**2))
    for i in range(itr):
        u1 = df.diffuse(u1, alpha, 0, D)
    return (u1*255).astype(int)


def RGBAKantBevGlatting(u, alpha=0.24, k=0.1, itr=1):
    timg = np.zeros(u.shape, int)  # Transformed image
    for i in range(u.shape[2]):
        timg[:, :, i] = BWKantBevGlatting(u[:, :, i], alpha, k, itr)
    return timg.astype(np.uint8)


class KantbevarendeGlatting(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Kantbevarende Glatting"
        self.function = RGBAKantBevGlatting
        self.params = [
            ("source", np.ndarray, None),
            ("alpha", float, 0.24),
            ("kantbevaring", float, 100),
            ("iterasjoner", int, 10)
        ]
        self.initDefaultValues()


if __name__ == "__main__":
    import imageio as im
    import matplotlib.pyplot as plt
    orig_im = im.imread("../testimages/raccoon.jpg")
    tImg = RGBAKantBevGlatting(orig_im, 0.24, 100, 5)
    plt.imshow(tImg)
    plt.show()
