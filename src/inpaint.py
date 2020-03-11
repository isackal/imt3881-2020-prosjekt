import imageio
import numpy as np
import matplotlib.pyplot as plt


orig_im = imageio.imread('../IOD.png')
mask = np.zeros((orig_im.shape[0], orig_im.shape[1]))
mask[45:75, 67:97] = 1
mask_im = orig_im[mask]
plt.imshow(mask_im)
plt.show(block=False)
input()
