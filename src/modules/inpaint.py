import numpy as np
import modifiers as md

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
    return 0.24*(y[0]+y[1]+y[2]+y[3]-4*y[4])


def inpaint(img, depth, mask):
    """
    Inpainter en valgt region av et bilde

    Parameters
    ----------
    img : np.ndarray
        Bilde med en region som skal inpaintes

    depth : int
        antall ganger diffusjonsligningen skal kjøres på bilde.
        Tallet ganges med 100 da en diffusjon utgjør liten forrandring

    masks : np.ndarray
         Maske over regionen som skal innpaintes.
    """
    # Potential improvements: enable non-square masks
    # Track every pixel that is true in mask ndarray and diffuse that
    img = img.astype(float) / 255
    maskCords = np.argwhere(mask)
    size = img.shape

    #   Generate view of image around inpaint region
    top = max(1, np.amin(maskCords[:, 0]))
    bottom = min(size[1]-1, np.amax(maskCords[:, 0]))
    left = max(1, np.amin(maskCords[:, 1]))
    right = min(size[0]-1, np.amax(maskCords[:, 1]))
    view = img[top:bottom, left:right]

    #   Generate diffrent views for diffusion. [top:bottom, left:right]
    mask = mask[top:bottom, left:right].astype(bool)
    lmask = np.roll(mask, -1, axis=0)
    rmask = np.roll(mask, 1, axis=0)
    tmask = np.roll(mask, -1, axis=1)
    bmask = np.roll(mask, 1, axis=1)

    print(mask.shape)

    for i in range(depth*100):
        a = view[lmask] + view[rmask] + view[tmask] + view[bmask] - 4*view[mask]
        view[mask] += a*0.24

    img[top:bottom, left:right] = view
    return (img * 255).astype(np.uint8)


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


if __name__ == "__main__":
    img = imageio.imread('../../test2.png')
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask[55:85, 200:225] = 1
    new_img = inpaint(img, 5, mask)
    plt.imshow(new_img)
    plt.show(block=True)
