import numpy as np
import modifiers as md
import cv2

from blurring import blurring
from colortogray import color_to_gray

#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def circularMask(w, h, epsilon=0.05):
    x, y = np.mgrid[0:w:1, 0:h:1]
    x = 2*x.astype(float)/(w-1) - 1
    y = 2*y.astype(float)/(h-1) - 1
    return x**2 + y**2 < 1+epsilon


def anonymisering(img):
    mask = np.zeros(img.shape[:2])
    mask = mask.astype(bool)
    gray = color_to_gray(img, 100, 0.1)
    xmlPath = "data/Haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(xmlPath)
    faces = face_cascade.detectMultiScale(gray, 1.02, 5)
    eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
    eyes = eyes[0:, :2]

    for (x, y) in eyes:
        mask[y, x] = True

    for (x, y) in eyes:
        view = mask[y-30:y+30, x-50:x+50]
        a = np.argwhere(view)
        if(a.shape[0] == 2):
            mask[y-30:y+70, x-10:x+10] = True

    """
    #Look for another nearby entry.
    while(eyes.any()):
        a = eyes[0]
        eyes = eyes[1:]
        for j in range(len(eyes)):
            if(a[0] >= eyes[j][0]-50 and a[0] <= eyes[j][0]+50):
                if(a[1] >= eyes[j][1]-50 and a[1] <= eyes[j][1]+50):
                    mask[a[1]:a[1]+50, a[0]:a[0]+50] = True
        #print(a)
    """
    #for (x, y, w, h) in faces:
        #mask[y:y+h, x:x+w] = circularMask(h, w)


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
    img = np.array(imageio.imread('../../../people.jpg'))
    new_img = anonymisering(img)

    plt.imshow(new_img)
    plt.show()
