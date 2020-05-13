import unittest
import numpy as np
import imageio

import anonymiser
import blurring
#import cloning
#import colortogray
#import contrast
import demosaic
import inpaint
#import kantBevGlatting


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
