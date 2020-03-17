import unittest
import numpy as np

from modules.inpaint import inpaint


class test_modul(unittest.TestCase):
    def test_inpaint(self):
        # Generer et "Bilde" som ska innpaintes
        img = np.zeros((11, 11)).astype(float)
        img[2:8, 2:8] = 1.0
        img[5, 5] = 0.0
        depth = 1
        masks = []
        masks.append(np.zeros((11, 11)))
        masks.append(np.zeros((11, 11)))
        masks.append(np.zeros((11, 11)))
        masks[0][2:6, 3:7] = 1
        masks[1][3:7, 3:7] = 1
        masks[2][4:8, 3:7] = 1
        img = inpaint(img, depth, masks)

        # Sjekk at diffusjon har gått inn til entry som var 0
        self.assertAlmostEqual(img[5, 5], 1.0)
