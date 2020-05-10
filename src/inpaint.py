import numpy as np
import modifiers as md
import poisson
import diffusion

#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def inpaint(img, itr, mask, alpha):
    """
    Blurs the image

    Alpha should remain below 0.25 to prevent numeric
    instablilty.

    Paramters
    ---------
    img : np.ndarray
        Source image
    depth : int
        Number of iterations (default = 50)
    mask : np.ndarray
        Region of image that should be inpainted
    Returns
    -------
    np.ndarray
        Inpainted image
    """
    if mask is None:  # No inpaint region specified, do no inpainting
        return img
    else:
        return diffusion.pre_diffuse(img, mask, alpha=alpha, itr=itr)


class Inpaint(md.Modifier):
    #   read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "Inpaint"
        self.function = inpaint
        self.params = [
            ("source", np.ndarray, None, md.FORMAT_RGBA),
            ("depth", int, 50),
            ("mask", np.ndarray, None, md.FORMAT_BOOL),
            ("alpha", float, 0.2)
        ]
        self.outputFormat = md.FORMAT_RGBA
        self.initDefaultValues()


#   Testfunksjon. Slett ved endelig release
if __name__ == "__main__":
    img = imageio.imread('../../test2.png').astype(float) / 255

    mask = np.zeros(img.shape[:2])
    mask[55:58, 207:213] = 1
    mask[58:61, 205:215] = 1
    mask[61:64, 203:218] = 1
    mask[64:68, 201:221] = 1
    mask[68:82, 200:225] = 1
    mask[82:85, 201:221] = 1
    mask[85:88, 203:218] = 1
    mask[88:91, 205:215] = 1
    mask[91:94, 207:213] = 1

    new_img = inpaint(img, 500, mask, 0.24)
    plt.imshow(new_img)
    plt.show()
