import unittest
import numpy as np
import imageio

import matplotlib.pyplot as plt

import anonymiser
import blurring
import cloning
import colortogray
import contrast
import demosaic
import inpaint
import kantBevGlatting


class test_modul(unittest.TestCase):
    def test_inpaint(self):
        # Generer et "Bilde" som ska innpaintes
        self.img = (np.ones((7, 7))).astype(float)
        self.img[3, 3] = 0
        self.itr = 1
        self.mask = np.zeros((7, 7))
        self.mask[1:6, 1:6] = 1
        self.alpha = 0.24

        self.img = inpaint.inpaint(self.img, self.itr, self.mask, self.alpha)

        self.assertNotAlmostEqual(self.img[3, 3], 0)
        self.assertAlmostEqual(self.img[0, 0], 1)

    def test_blurring(self):
        # Generer et "Bilde" som ska blurres
        self.img = (np.ones((7, 7))).astype(float)
        self.img[3, 3] = 0
        self.itr = 1
        self.mask = np.zeros((7, 7))
        self.mask[1:6, 1:6] = 1
        self.alpha = 0.24

        self.img = blurring.blurring(self.img, self.itr, self.mask, self.alpha)

        self.assertNotAlmostEqual(self.img[3, 3], 0)
        self.assertAlmostEqual(self.img[0, 0], 1)

    def test_demosaic(self):
        self.red = np.zeros((7, 7)).astype(float)
        self.green = np.zeros((7, 7)).astype(float)
        self.blue = np.zeros((7, 7)).astype(float)
        self.red[0::2, 0::2] = 1
        self.green[1::2, 0::2] = 1
        self.green[0::2, 1::2] = 1
        self.blue[1::2, 1::2] = 1
        self.img = demosaic.demosaic(self.red, self.green, self.blue)

        # Check RGB channel that was previously 0
        self.assertNotAlmostEqual(self.img[1, 0, 0], 0)
        self.assertNotAlmostEqual(self.img[0, 0, 1], 0)
        self.assertNotAlmostEqual(self.img[0, 0, 2], 0)

    def test_anonymisering(self):
        self.img = (np.ones((20, 20, 3))).astype(float)
        self.itr = 1
        self.alpha = 0.24

        try:  # Expected crash as no face present
            anonymiser.anonymisering(self.img, self.itr, self.alpha)
            self.assertEqual(1, 0)
        except ValueError:  # Expected error
            self.assertEqual(1, 1)
        except Exception as e:  # Wrong error
            print(e)
            self.assertEqual(1, 0)

        self.img = np.asarray(
                imageio.imread('../testimages/Anon_eye1.png')
            ).astype(float) / 255
        try:  # expected to run as there is a face
            anonymiser.anonymisering(self.img, self.itr, self.alpha)
            self.assertEqual(1, 1)
        except Exception as e:  # Did not create a boolean mask anywhere
            print(e)
            self.assertEqual(1, 0)

    def test_colorToGray(self):
        self.img = np.ones((7, 7, 3)).astype(float)

        self.img = colortogray.color_to_gray(self.img)

        # Test if multiple channels are deleted
        self.assertEqual(len(self.img.shape), 2)

    def test_kantBevGlatting(self):
        self.img = np.zeros((11, 11, 3)).astype(float)

        self.img[4:7, :] = 1
        self.img[5, :] = 0.5
        self.img[3, :] = 0.5
        self.img[7, :] = 0.5
        self.alpha = 0.24
        self.k = 100000
        self.itr = 5

        self.blurImg = np.copy(self.img)
        self.blurImg = blurring.blurring(
            self.blurImg, self.itr, None, self.alpha
        )

        self.img = kantBevGlatting.RGBAKantBevGlatting(
            self.img, self.alpha, self.k, self.itr
        )

        self.sum = np.sum(self.img - self.blurImg)
        self.assertNotAlmostEqual(self.sum, 0)

    def test_kontrast(self):
        self.img = np.zeros((11, 11, 3)).astype(float)

        self.img[4:7, :] = 1
        self.img[5, :] = 0.5
        self.img[3, :] = 0.5
        self.img[7, :] = 0.5

        self.orig_img = np.copy(self.img)
        self.img = contrast.contrast(self.img)

        self.sum = np.sum(self.img - self.orig_img)
        self.assertNotAlmostEqual(self.sum, 0)

    def test_cloning(self):
        self.img1 = np.zeros((11, 11, 3)).astype(float)
        self.img2 = np.zeros((11, 11, 3)).astype(float)
        self.img2[2:4, 2:4] = 1
        self.mask1 = np.copy(self.img2).astype(bool)
        self.itr = 5
        self.alpha = 0.24

        self.img = cloning.cloning(
            self.img1, self.img2, self.itr, self.mask1, None, self.alpha
        )

        self.sum = np.sum(self.img)
        self.assertNotAlmostEqual(self.sum, 0)

        # Just to run the extra line in else statement, no real test
        try:
            self.img = cloning.cloning(
                self.img1, self.img2, self.itr,
                self.mask1, self.mask1, self.alpha
            )
        except Exception as e:
            print(e)
            self.assertEqual(1, 0)
