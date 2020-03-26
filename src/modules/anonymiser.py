import numpy as np
import modifiers as md
import cv2

from modules.blurring import blurring
from modules.colortogray import color_to_gray
#   Brukt for testfunksjon. Slett ved endelig release
import imageio
import matplotlib.pyplot as plt


def anonymisering(img):
    mask = imageio.imread('../../People_binary_mask.jpg').astype(bool)
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
    img = cv2.imread("../../People.jpg")
    print(type(img))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)

    cv2.imwrite('mask1.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    img = np.array(imageio.imread('../../People.jpg'))
    new_img = anonymisering(img)
    
    plt.imshow(new_img)
    plt.show()
