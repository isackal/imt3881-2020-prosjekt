import numpy as np
import modifiers as md


def color_to_gray(img, n, alpha):
    """
    Converts image to grayscale

    Paramters
    ---------
    img : np.ndarray
        Source image
    n : int
        Number of iterations (default = 100)
    alpha : float
        delta_t / delta_x**2 (default = 0.1)

    Returns
    -------
    np.ndarray
        Grayscale image
    """
    og_img = img[:, :, :3].astype(float) / 255

    # Length of color vector
    rgb_len = np.sum(og_img**2, axis=2)**0.5

    # Find length of gradient
    grad_len = np.ones((og_img.shape[0], og_img.shape[1]))

    grad_len[1:, 1:] = np.sqrt(2 * rgb_len[1:, 1:] * (
                               rgb_len[1:, 1:] -
                               rgb_len[:-1, 1:] -
                               rgb_len[1:, :-1]
                               ) +
                               rgb_len[:-1, 1:]**2 +
                               rgb_len[1:, :-1]**2) / np.sqrt(3)
    grad_len[0, :] = grad_len[1, :]
    grad_len[:, 0] = grad_len[:, 1]

    # Find direction
    grad = np.ones((og_img.shape[0], og_img.shape[1], 2))
    grad[1:, 1:, 0] = (np.sum(og_img[1:, 1:], axis=2) -
                       np.sum(og_img[:-1, 1:], axis=2))
    grad[1:, 1:, 1] = (np.sum(og_img[1:, 1:], axis=2) -
                       np.sum(og_img[1:, :-1], axis=2))

    unit_len = np.sum(grad[1:, 1:]**2, axis=2)**0.5
    unit_len[unit_len == 0] = -1  # Prevent division by zero

    # Complete gradient by setting length to previously found length
    grad_len[1:, 1:] *= 1 / unit_len
    grad_len[grad_len < 0] = 0  # Prevent division by zero
    grad[1:, 1:, 0] *= grad_len[1:, 1:]
    grad[1:, 1:, 1] *= grad_len[1:, 1:]
    grad[0, :] = grad[1, :]
    grad[:, 0] = grad[:, 1]

    # h function is the gradient of grad
    h = np.ones((og_img.shape[0], og_img.shape[1]))
    h[1:, 1:] = (2 * np.sum(grad[1:, 1:], axis=2) -
                 np.sum(grad[:-1, 1:], axis=2) - np.sum(grad[1:, :-1], axis=2))
    h[0, :] = h[1, :]
    h[:, 0] = h[:, 1]

    # Init value, avrage of the three colors
    new_img = np.sum(og_img, axis=2) / 3

    for i in range(n):
        laplace = (new_img[2:, 1:-1] +
                   new_img[:-2, 1:-1] +
                   new_img[1:-1, 2:] +
                   new_img[1:-1, :-2] -
                   4 * new_img[1:-1, 1:-1])

        new_img[1:-1, 1:-1] += alpha * (laplace - h[1:-1, 1:-1])
        new_img[0, :] = new_img[1, :]
        new_img[-1, :] = new_img[-2, :]
        new_img[:, 0] = new_img[:, 1]
        new_img[:, -1] = new_img[:, -2]

        # Trim values outside of range
        new_img[new_img > 1] = 1
        new_img[new_img < 0] = 0

    new_img = (new_img * 255).astype(np.uint8)
    alpha_channel = np.full(new_img.shape, 255, dtype=np.uint8)

    return np.dstack((new_img, new_img, new_img, alpha_channel))


class Colortogray(md.Modifier):
    def __init__(self):
        super().__init__()
        self.name = "color to gray"
        self.function = color_to_gray
        self.params = [
            ("img", np.ndarray, None),
            ("iterations", int, 100),
            ("alpha", float, 0.1)
        ]
        self.initDefaultValues()
