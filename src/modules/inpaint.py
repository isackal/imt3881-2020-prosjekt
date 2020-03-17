import numpy as np
import modifiers as md


def diffusion(y):
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
    return 0.49*(y[2]-2*y[1]+y[0])


def inpaint(img, depth, masks):
    """
    Inpainter en valgt region av et bilde

    Parameters
    ----------
    img : np.ndarray
        Bilde med en region som skal inpaintes

    depth : int
        antall ganger diffusjonsligningen skal kjøres på bilde.
        Tallet ganges med 100 da en diffusjon utgjør liten forrandring

    masks : np.ndarray(boolean)
        3 masker med litt forskjellige views rund regionen som skal innpaintes.
    """
    for i in range(len(masks)):
        # convert masks into boolean masks for np.ndarray view compatability
        masks[i] = masks[i].astype(bool)

    for i in range(100*depth):
        views = np.array([img[masks[0]], img[masks[1]], img[masks[2]]])
        img[masks[1]] += diffusion(views)
    return img


class Inpaint(md.Modifier):
    # read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "Inpaint"
        self.function = inpaint
        self.params = [
            ("source", np.ndarray, None),
            ("depth", int, 1),
            ("masks", list, None)
        ]
        self.initDefaultValues()


# Default values saved for possible later testing useage
"""
if __name__ == "__main__":
    u = imageio.imread('../IOD.png').astype(float)/255
    masks = []
    masks.append(np.zeros((u.shape[0], u.shape[1])).astype(bool))
    masks.append(np.zeros((u.shape[0], u.shape[1])).astype(bool))
    masks.append(np.zeros((u.shape[0], u.shape[1])).astype(bool))
    masks[0][45:75, 66:96] = True
    masks[1][45:75, 67:97] = True
    masks[2][45:75, 68:98] = True
    inpaint(u, 5, masks)
"""
