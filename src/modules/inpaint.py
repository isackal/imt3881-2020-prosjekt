import numpy as np
import modifiers as md
import sys
np.set_printoptions(threshold=sys.maxsize)
#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def diffusjon(y):
    """
    Kjører diffusjon inn i regionen som ønskes inpainted

    Bruker K = 0.49 som konstant utenfor da dette øker
    hastigheten til diffusjon inn i regionen.
    og hindrer mulig numerisk ustabilitet ved k > 0.5

    Parameters
    ----------
    y : np.ndarray
        3 arrays med litt forskjellige views ved regionen som skal innpaintes

    Returns
    -------
    np.ndarray
        regionen etter en diffusjon
    """
    #return 0.24*(y[0]+y[1]+y[2]+y[3]-4*y[4]) # Speiler
    #return 0.24*y[0] # Speiler
    #return 0.24*y[1] # Speiler
    #return 0.24*y[2] # Speiler ikke
    #return 0.24*y[3] # Speiler ikke
    #return 0.24*y[4] # Speiler ikke
    return 0.49*(y[0] + y[1] - 2 * y[2])


def inpaint(img, depth, mask):
    img = img.astype(float) / 255
    mask = mask.astype(bool)
    maskCords = np.argwhere(mask)
    size = img.shape
    #Issue: Diffuses all values besides the edges.

    #   Lag et view av regionen som skal innpaintes
    top = np.amin(maskCords[:, 0])
    bottom = np.amax(maskCords[:, 0])
    left = np.amin(maskCords[:, 1])
    right = np.amax(maskCords[:, 1])

    view = img[top:bottom, left:right]
    viewMask = mask[top:bottom, left:right]
    t_viewMask = np.roll(viewMask, -1, axis=0)
    b_viewMask = np.roll(viewMask, 1, axis=0)

    for i in range(depth):
        views = np.array([
            #view[t_viewMask],
            #view[b_viewMask],
            view[l_viewMask],
            view[r_viewMask],

            view[viewMask]
        ])
        view[viewMask] += diffusjon(views)

    return img


class Inpaint(md.Modifier):
    #   read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "Inpaint"
        self.function = inpaint
        self.params = [
            ("source", np.ndarray, None),
            ("depth", int, 1),
            ("mask", np.ndarray, None)
        ]
        self.initDefaultValues()


#   Testfunksjon. Slett ved endelig release
if __name__ == "__main__":
    img = imageio.imread('../../test2.png')
    mask = np.ones((img.shape[0], img.shape[1]))
    red = np.zeros(img.shape[:2])
    mask[::2, ::2] = 0
    red[::2, ::2] = img[::2, ::2, 0]
    """
    mask[55:58, 207:213] = 1
    mask[58:61, 205:215] = 1
    mask[61:64, 203:218] = 1
    mask[64:68, 201:221] = 1
    mask[68:82, 200:225] = 1
    mask[82:85, 201:221] = 1
    mask[85:88, 203:218] = 1
    mask[88:91, 205:215] = 1
    mask[91:94, 207:213] = 1
    """
    new_img = inpaint(red, 50, mask)
    plt.imshow(red, cmap=plt.cm.gray)
    plt.show()
    plt.imshow(new_img, cmap=plt.cm.gray)
    plt.show()

