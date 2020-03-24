import unittest
import numpy as np

from modules.inpaint import inpaint
from modules.demosaic import demosaic


class test_modul(unittest.TestCase):
    def test_inpaint(self):
        # Generer et "Bilde" som ska innpaintes
        self.img = (np.ones((11, 11)) * 255).astype(np.uint8)
        self.img[5, 5] = 0
        self.depth = 500
        self.mask = np.zeros((11, 11))
        self.mask[3:7, 3:7] = 1
        self.img = inpaint(self.img, self.depth, self.mask)

        # Sjekk at diffusjon har gått inn til entry som var 0
        self.assertAlmostEqual(self.img[5, 5], 255)

    def test_demosaic(self):
        self.red = np.zeros((4, 4))
        self.blue = np.zeros((4, 4))
        self.green = np.zeros((4, 4))
        self.red[::2, ::2] = 255
        self.blue[1::2, 1::2] = 255
        self.green[1::2, ::2] = 255
        self.green[::2, 1::2] = 255
        self.red = self.red.astype(np.uint8)
        self.blue = self.blue.astype(np.uint8)
        self.green = self.green.astype(np.uint8)

        self.img = demosaic(self.red, self.blue, self.green)
        print(self.img)
        self.img = self.img.astype(bool)
        nonDemosaic = np.argwhere(self.img)
        self.assertTrue(nonDemosaic.all())
