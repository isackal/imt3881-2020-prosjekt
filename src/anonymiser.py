import numpy as np
import cv2 as cv

import modifiers as md
from colortogray import color_to_gray
import diffusion


def circularMask(w, h, epsilon=0.05):
    """
    Creates a circular binary mask

    Parameters
    ----------
    w : int
        width of the circle

    h : int
        height of the circle

    epsilon : float
        Tolerance, higher number accepts a larger region

    Returns
    -------
    np.ndarray (w*h)
        Bolean array with true values in a circle
    """
    x, y = np.mgrid[0:w:1, 0:h:1]
    x = 2*x.astype(float)/(w-1) - 1
    y = 2*y.astype(float)/(h-1) - 1
    return x**2 + y**2 < 1+epsilon


def anonymisering(img, itr, alpha):
    """
    Creates a binary mask for where to blur an image

    Looks for faces and eyes in images and creates binary masks
    That blurring modules can blur out

    Paramters
    ---------
    img : np.ndarray
        Source image

    Returns
    -------
    np.ndarray
        Anonymized image
    """
    # Create a binary mask in the shape of the image
    mask = np.zeros(img.shape[:2]).astype(bool)
    # Converted to grayscale for face detection
    gray = color_to_gray(img, itr=10)
    gray = (gray * 255).astype(np.uint8)

    # Using already trained ML algorithm as basis for face and eye detection
    eye_cascade = cv.CascadeClassifier('data/haarcascade_eye.xml')
    face_cascade = cv.CascadeClassifier('data/Haarcascade_frontalface_alt.xml')

    # Find position and size of eyes and faces in the image as np.ndarray
    faces = face_cascade.detectMultiScale(gray, 1.02, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.02, 5)
    size = mask.shape[:2]

    # If a face is found, create a blurring mask in that region.
    # Some chance of false positives, but priority on blurring to much
    # More important to ensure everything that needs to be blurred is blurred.

    for (x, y, w, h) in faces:
        mask[y:y+int(1.1*h), x:x+w] += circularMask(int(1.1*h), w)

    # Mark all places where ML algorithm think there is an eye
    for (x, y, w, h) in eyes:
        mask[y, x] = True

    # Looks for 2 eyes in a close region to eachother, increases Recall
    # (Less blurring of regions which are not actually faces)
    for (x, y, w, h) in eyes:
        # Create a rectangle around a potential eye
        top = int(max(0, y-(0.2*h)))
        bottom = int(min(size[0], y+h))
        left = int(max(0, x-(1.2*w)))
        right = int(min(size[1], x+(2.2*w)))

        # Find out how many values are true in the region
        eyesDetected = np.argwhere(mask[top:bottom, left:right])

        # If it found another eye in the region create a mask to anonymize
        if(eyesDetected.shape[0] == 2):

            # Find midpoint between the eyes
            top += int((eyesDetected[0, 0] + eyesDetected[1, 0])*.5)
            left += int((eyesDetected[0, 1] + eyesDetected[1, 1])*.5)

            # Create region around midpoint of eyes where blurring should occur
            bottom = min(size[0], top+3*h)
            top = max(0, top-h)
            right = min(size[1], left+3*w)
            left = max(0, left-w)

            # Delta height and width
            dh = bottom-top
            dw = right-left

            # create blurring mask. Larger than for face blurring above
            # to ensure whole face is blurred
            mask[top:bottom, left:right] += circularMask(dh, dw)

        # Skip regions where a mask is already created
        # Also prevents some false positives
        elif(eyesDetected.shape[0] > 2):
            pass

        # If no other eye found, prevent region from being blurred at all
        else:
            mask[y, x] = False

    # Return image after a blurring process is run in regions where faces are.

    return diffusion.pre_diffuse(img, mask, alpha=alpha, itr=itr)


class Anonymisering(md.Modifier):
    #   read usage in ../modifiers.py
    def __init__(self):
        super().__init__()
        self.name = "Anonymisering"
        self.function = anonymisering
        self.params = [
            ("img", np.ndarray, None),
            ("itr", int, 100),
            ("alpha", float, 0.24)
        ]
        self.initDefaultValues()
