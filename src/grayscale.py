import numpy as np
import diffusion as df
import modifiers as md
import imageMath as imth


def colorToGray(u, alpha=0.24, itr=10, _k=1.):
    u0 = u.astype(float)/255
    g = np.zeros(u[:, :, 0].shape)  # Gradienten lengde
    vx = g+0.0001  # Retningen i x retning
    vy = g+0.0001  # Retningen i y retning
    _max = np.sqrt(6)  # sqrt(3) if 3 channels (RGB)

    for i in range(3):
        _gX = df.gX(u0[:, :, i])
        _gY = df.gY(u0[:, :, i])
        g += _gX**2 + _gY**2
    colSum = np.sum(u0[:, :, :3], axis=2)
    vx = df.gX(colSum)
    vy = df.gY(colSum)
    g = np.sqrt(g)/_max
    k = g / np.sqrt(vx**2 + vy**2)  # brukers for a tilpasse lengden.
    vx = vx * k
    vy = vy * k
    h = _k*(vx + vy)  # divergensen av g
    u1 = np.sum(u0[:, :, :3], axis=2) / 3
    # u1 = np.sqrt(np.sum(u0**2, axis=2))/_max
    for i in range(itr):
        u1 = df.diffuse(u1, alpha, -h)
    return imth.grayToRGBA(np.clip((u1*255),0,255).astype(np.uint8))


class CTG(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "CTG"
        self.function = colorToGray
        self.params = (
            ("source", np.ndarray, None),
            ("Alpha", float, 0.2),
            ("Iterations", int, 5),
            ("K", float, 1)
        )
        self.initDefaultValues()

