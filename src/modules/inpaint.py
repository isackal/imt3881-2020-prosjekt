import numpy as np
import modifiers as md

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
    img = img.astype(float) / 255
    mask = mask.astype(bool)
    maskCords = np.argwhere(mask)
    size = img.shape

    #   Lag et view av regionen som skal innpaintes
    top = max(2, np.amin(maskCords[:, 0])) - 1
    bottom = min(size[0]-3, np.amax(maskCords[:, 0])) + 2
    left = max(2, np.amin(maskCords[:, 1])) - 1
    right = min(size[1]-3, np.amax(maskCords[:, 1])) + 2

    view = img[top:bottom, left:right]
    viewMask = mask[top:bottom, left:right]

    #   Lag forskjellige masker for view regionen
    t_viewMask = np.roll(viewMask, -1, axis=0)
    b_viewMask = np.roll(viewMask, 1, axis=0)
    l_viewMask = np.roll(viewMask, -1, axis=1)
    r_viewMask = np.roll(viewMask, 1, axis=1)

    #   Diffuser fargene rundt regionen inn i regionen
    for i in range(depth):
        views = np.array([
            #view[t_viewMask],
            #view[b_viewMask],
            view[l_viewMask],
            view[r_viewMask],
            view[viewMask]
        ])
        view[viewMask] += diffusjon(views)

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


#   Testfunksjon. Slett ved endelig release
if __name__ == "__main__":
    img = imageio.imread('../../test2.png')
    mask = np.zeros((img.shape[0], img.shape[1]))
    mask[55:58, 207:213] = 1
    mask[58:61, 205:215] = 1
    mask[61:64, 203:218] = 1
    mask[64:68, 201:221] = 1
    mask[68:82, 200:225] = 1
    mask[82:85, 201:221] = 1
    mask[85:88, 203:218] = 1
    mask[88:91, 205:215] = 1
    mask[91:94, 207:213] = 1
    new_img = inpaint(img, 15, mask)
    plt.imshow(new_img)
    plt.show()
