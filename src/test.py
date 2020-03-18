import unittest
import numpy as np

from modules.inpaint import inpaint


class test_modul(unittest.TestCase):
    def test_inpaint(self):
        # Generer et "Bilde" som ska innpaintes
        img = (np.ones((11, 11)) * 255).astype(np.uint8)
        img[5, 5] = 0
        depth = 5
        mask = np.zeros((11, 11))
        mask[3:7, 3:7] = 1
        img = inpaint(img, depth, mask)

        # Sjekk at diffusjon har gått inn til entry som var 0
        self.assertAlmostEqual(img[5, 5], 255)
