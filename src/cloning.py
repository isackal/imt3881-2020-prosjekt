import modifiers as md
import numpy as np


def cloning(img1, img2, n, mask1, mask2, alpha):
    """
    Clones a portion of img2 into img1

    Both images and masks needs to be the same size, but
    the masks position can be different. If only one
    mask is specified it will use mask1 on both images.
    Needs many iteration to achieve a decent result.

    Paramters
    ---------
    img1 : np.ndarray
        Source image
    img2 : np.ndarray
        Clone source image
    n : int
        Number of iterations (default = 100)
    mask1 : np.ndarray
        Region of the source image to be cloned into
    mask2 : np.ndarray
        Region of the clone source to clone from. Region
        must be the same size as mask1 (default = None)
    alpha : float
        delta_t / delta_x**2 (default = 0.24)

    Returns
    -------
    np.ndarray
        Image with amplified contrast
    """
    img = np.copy(img1)
    clone_source = np.copy(img2)
    new_img = img

    # Make a list with the two masks
    mask = list()
    # mask1 = mask1.astype(bool)

    # If only one mask is recieved, clone from the same coords
    if mask2 is None:
        mask = [mask1, mask1]
    else:
        # mask2 = mask2.astype(bool)
        mask = [mask1, mask2]

    # Clear borders of the mask to prevent wrapping
    for m in mask:
        m[0, :] = False
        m[-1, :] = False
        m[:, 0] = False
        m[:, -1] = False

    h = (clone_source[np.roll(mask[1], -1, axis=0)] +
         clone_source[np.roll(mask[1], 1, axis=0)] +
         clone_source[np.roll(mask[1], -1, axis=1)] +
         clone_source[np.roll(mask[1], 1, axis=1)] -
         4 * clone_source[mask[1]])

    for i in range(n):
        laplace = (new_img[np.roll(mask[0], -1, axis=0)] +
                   new_img[np.roll(mask[0], 1, axis=0)] +
                   new_img[np.roll(mask[0], -1, axis=1)] +
                   new_img[np.roll(mask[0], 1, axis=1)] -
                   4 * new_img[mask[0]])

        new_img[mask1] += alpha * (laplace - h)
        new_img[~mask1] = img[~mask1]

    # Trim values outside of range
    new_img[new_img > 1] = 1
    new_img[new_img < 0] = 0

    # Return the correct format
    # new_img = (new_img * 255).astype(np.uint8)
    # alpha_channel = np.full((mask1.shape[0], mask1.shape[1], 1),
    #                        255, dtype=np.uint8)

    return new_img


class Cloning(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Cloning"
        self.function = cloning
        self.params = [
            ("img1", np.ndarray, None, md.FORMAT_RGB),
            ("img2", np.ndarray, None, md.FORMAT_RGB),
            ("iterations", int, 100),
            ("mask1", np.ndarray, None, md.FORMAT_BOOL),
            ("mask2", np.ndarray, None, md.FORMAT_BOOL),
            ("alpha", float, 0.24)
        ]
        self.outputFormat = md.FORMAT_RGB
        self.initDefaultValues()
