import numpy as np
import modifiers as md


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
    left = maskCords[0][1]
    right = maskCords[-1][1]
    top = maskCords[0][0]
    bottom = maskCords[-1][0]
    size = img.shape
    i = 0
    while i < depth*100:
        i += 1
        views = np.array([
            img[top:bottom, max(1, left-1):right-1],
            img[top:bottom, left+1:min(size[1]-1, right+1)],
            img[max(1, top-1):bottom-1, left:right],
            img[top+1:min(size[0]-1, bottom+1), left:right],
            img[top:bottom, left:right]
            ])
        img[top:bottom, left:right] += diffusjon(views)
    return (img * 255).astype(np.uint8)


class Inpaint(md.Modifier):
    # read usage in ../modifiers.py
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
