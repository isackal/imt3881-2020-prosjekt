import imageio
import numpy as np
import matplotlib.pyplot as plt


def f1(y):
    return 0.49*(y[2]-2*y[1]+y[0])


def inpaint(img, depth, masks):
    for i in range(100*depth):
        views = np.array([img[masks[0]], img[masks[1]], img[masks[2]]])
        img[masks[1]] += f1(views)
    plt.imshow(img)
    plt.show(block=True)


if __name__ == "__main__":
    u = imageio.imread('../IOD.png').astype(float)/255
    masks = []
    masks.append(np.zeros((u.shape[0], u.shape[1])).astype(bool))
    masks.append(np.zeros((u.shape[0], u.shape[1])).astype(bool))
    masks.append(np.zeros((u.shape[0], u.shape[1])).astype(bool))
    masks[0][45:75, 66:96] = True
    masks[1][45:75, 67:97] = True
    masks[2][45:75, 68:98] = True
    inpaint(u, 5, masks)

