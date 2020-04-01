import numpy as np
import modifiers as md
import cv2

import sys
np.set_printoptions(threshold=sys.maxsize)


from modules.blurring import blurring
from modules.colortogray import color_to_gray
#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def anonymisering(img):
    mask = np.zeros(img.shape[:2])
    mask = mask.astype(bool)
    gray = color_to_gray(img, 100, 0.1)
    xmlPath = "data/Haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(xmlPath)
    faces = face_cascade.detectMultiScale(gray, 1.05, 3)
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in faces:
        mask[y:y+h, x:x+w] = True
#    for (x, y, w, h) in eyes:
#        mask[y:y+h, x:x+w] = True
    #print(mask)
    return blurring(img, 1, 0.24, mask)


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
    img = np.array(imageio.imread('../../People3.jpg'))
    new_img = anonymisering(img)

    plt.imshow(new_img)
    plt.show()
