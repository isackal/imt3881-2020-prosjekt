import modifiers as md
import numpy as np
import poisson
import diffusion

#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt

def blurring(img, n, alpha, mask):
    """
    Blurs the image

    Alpha should remain below 0.24 to prevent numeric
    instablilty.

    Paramters
    ---------
    img : np.ndarray
        Source image
    n : int
        Number of iterations (default = 10)
    alpha : float
        delta_t / delta_x**2 (default = 0.24)

    Returns
    -------
    np.ndarray
        Blurred image
    """
    if mask is None:  # Blur whole image if no mask is given
        mask = np.ones(img.shape[:2])

    
    #for i in range(3):
    #    img[:, :, i] = diffusion.pre_diffuse(img[:, :, i], mask, 'e', 'n', alpha, n, 0, 1.)

    return diffusion.pre_diffuse(img, mask, 'e', 'n', alpha, n, 0, 1.)
    #return poisson.implisitt(img, n, mask, alpha)

class Blurring(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "Blurring"
        self.function = blurring
        self.params = [
            ("img", np.ndarray, None),
            ("iterations", int, 10),
            ("alpha", float, 0.24),
            ("mask", np.ndarray, None)
        ]
        self.initDefaultValues()


#   Testfunksjon. Slett ved endelig release
if __name__ == "__main__":
    img = imageio.imread('../../face.png').astype(float) / 255

    mask = np.zeros(img.shape[:2])
    mask[50:250, 50:250] = 1

    new_img = blurring(img, 50, 0.1, mask)
    plt.imshow(new_img)
    plt.show()