import numpy as np
import modifiers as md
import sys
np.set_printoptions(threshold=sys.maxsize)
#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def inpaint(img, depth, mask):
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

    new_img = img.astype(float) / 255
    img = img.astype(float) / 255

    mask = mask.astype(bool)
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    t_mask = np.roll(mask, -1, axis=0)
    b_mask = np.roll(mask, 1, axis=0)
    l_mask = np.roll(mask, -1, axis=1)
    r_mask = np.roll(mask, 1, axis=1)

    for i in range(depth):
        laplace = (new_img[t_mask] +
                   new_img[b_mask] +
                   new_img[l_mask] +
                   new_img[r_mask] -
                   4 * new_img[mask]
                   )
        new_img[mask] += 0.24 * laplace
        new_img[~mask] = img[~mask]

    return (new_img * 255).astype(np.uint8)


class Inpaint(md.Modifier):
    #   read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "Inpaint"
        self.function = inpaint
        self.params = [
            ("source", np.ndarray, None),
            ("depth", int, 50),
            ("mask", np.ndarray, None)
        ]
        self.initDefaultValues()


#   Testfunksjon. Slett ved endelig release
if __name__ == "__main__":
    img = imageio.imread('../../test2.png')

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

    new_img = inpaint(img, 100, mask)
    plt.imshow(new_img, cmap=plt.cm.gray)
    plt.show()