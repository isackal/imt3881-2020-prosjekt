import numpy as np
import diffusion as df


def colorToGray(u, alpha=0.24, itr=10, _k=1.):
    u0 = u.astype(float)/255
    g = np.zeros(u[:, :, 0].shape)  # Gradienten lengde
    vx = g+0.0001  # Retningen i x retning
    vy = g+0.0001  # Retningen i y retning
    _max = np.sqrt(1.*u.shape[2])  # sqrt(3) if 3 channels (RGB)

    for i in range(u.shape[2]):
        _gX = df.gX(u0[:, :, i])
        _gY = df.gY(u0[:, :, i])
        vx += _gX
        vy += _gY
        g += _gX**2 + _gY**2
    g = np.sqrt(g)/_max
    k = g / np.sqrt(vx**2 + vy**2)  # brukers for a tilpasse lengden.
    vx = vx * k
    vy = vy * k
    h = _k*(vx + vy)  # divergensen av g
    u1 = np.sum(u0, axis=2) / u0.shape[2]
    # u1 = np.sqrt(np.sum(u0**2, axis=2))/_max
    for i in range(itr):
        u1 = df.diffuse(u1, alpha, -h)
    return u1
