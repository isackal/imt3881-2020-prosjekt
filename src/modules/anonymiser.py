import numpy as np
import modifiers as md
import cv2 as cv

from modules.blurring import blurring

#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def anonymisering(img):
    mask = imageio.imread('../../People_binary_mask.jpg').astype(bool)
    """
    mask = np.zeros((img.shape[:2]))
    mask[10:125, 50:130] = 1
    mask[10:125, 400:480] = 1"""
    print(cv.__version__)
    return blurring(img, 750, 0.24, mask)


class Anonymisering(md.Modifier):
    #   read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "Anonymisering"
        self.function = anonymisering
        self.params = [
            ("img", np.ndarray, None)
        ]
        self.initDefaultValues()


#   Testfunksjon. Slett ved endelig release
if __name__ == "__main__":
    img = imageio.imread('../../People.jpg')
    new_img = anonymisering(img)
    plt.imshow(new_img)
    plt.show()
