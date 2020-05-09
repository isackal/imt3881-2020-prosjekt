import numpy as np
import random
import erh as eh  # for error handling
import os
import imageio as im
import matplotlib.pyplot as plt


Z_MIN = 0
Z_MAX = 255

def loadImages(folder):
    """
    Load images from a folder and returns them.
    the exposure time dt is part of the file name.
    """
    images = []
    dt = []
    files = os.listdir(folder)
    for file in files:
        if ".png" not in file:
            continue
        _fl = folder+"/"+file
        print(_fl)
        img = im.imread(folder+"/"+file)
        _dt = 0
        _mod = 1
        for char in file[::-1][4:]:
            if char == '_':
                break
            _dt += (ord(char)-ord('0')) * _mod
            _mod *= 10
        images.append(img)
        dt.append(_dt)
    # Sort:
    for i in range(1,len(dt)):
        _img = images[i]
        _dt = dt[i]
        j = i
        while (j>0 and (dt[j-1]>_dt)):
            dt[j] = dt[j-1]
            images[j] = images[j-1]
            j -= 1
        dt[j]=_dt
        images[j]=_img
    
    return (images, dt)


def getZT(imgDtPairs, amount, seed=23):
    """
    Get Z

    Parameters
    ----------
    imgDtPairs : pair of two lists, images and delta times
    amount     : between 0 and 1, selets random samples proportional to this parameter.
    Select pixel samples from the images and return z
    """
    # Z = np.zeros((sampleSize, 3, len(imgDtPairs[0])))
    np.random.seed(seed)
    _mask = np.random.rand(*list(imgDtPairs[0][0].shape[:2]))
    __mask = None
    _sum = 0
    epsilon = 0.000001
    pmin=0.
    pmax=100.
    while ((_sum != amount) and ((pmax-pmin)**2 > epsilon**2)):
        pmid = (pmin + pmax)/2
        __mask = _mask <= pmid
        _sum = np.sum(__mask)
        if _sum > amount:
            pmax = pmid
        else:
            pmin = pmid
    Z = np.zeros((_sum, len(imgDtPairs[1]), 3), np.int32)
    print(Z.shape)
    dt = np.zeros(len(imgDtPairs[1]))
    for i in range(len(imgDtPairs[0])):
        _img = imgDtPairs[0][i]
        Z[:, i, 0] = _img[:, :, 0][__mask]
        Z[:, i, 1] = _img[:, :, 1][__mask]
        Z[:, i, 2] = _img[:, :, 2][__mask]
        dt[i] = imgDtPairs[1][i]
    return Z, dt


def w(pixel_value, z_min=Z_MIN, z_max=Z_MAX):
    """
    Weighting function

    Parameters
    ----------
    pixel_value : float

    Returns
    -------
    float
        Weighted pixel value
    """
    if pixel_value <= (0.5 * (z_min + z_max)):
        return pixel_value - z_min
    return z_max - pixel_value


def objective(Z, B, l=0.2):
    """
    Finds response function g and log irradiance values

    Given a set of pixel values observed for several pixels in
    several images with different exposure times. This functions will find
    the response function g as well as the log film irradiance vales.

    Parameters
    ----------
    Z[i, j] : <numpy.ndarray>
        The pixel values of pixel location number i in image j
    B[j] : <numpy.ndarray>
        Array with log shutter speed for image j
    l : float
        Constant that determines the amount of smoothness

    Returns
    -------
    g[z] : <numpy.ndarray>
        Log exposure corresponding to pixel value z
    lE[i] : <numpy.ndarray>
        Log film irradiance at pixel location i
    """
    n = 256

    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, Z.shape[0] + n))
    print(A.shape)
    b = np.zeros(A.shape[0])

    # Include the data-fitting equations

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            w_ij = w(Z[i, j] + 1)
            A[k, Z[i, j]] = w_ij
            A[k, n + i] = -w_ij
            b[k] = w_ij * B[j]
            k += 1

    # Fix the curve by setting its middle value to 0

    A[k, 128] = 1
    k += 1

    # Include the smoothness equations

    for i in range(n - 2):
        A[k, i] = l * w(i + 1)
        A[k, i + 1] = -2 * l * w(i + 1)
        A[k, i + 2] = l * w(i + 1)
        k += 1

    # Solve the system
    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, b)

    g = x[0:n]
    lE = x[n:]

    return g, lE


def reconstruct(itp, g, channel=0):
    lnE = np.zeros(itp[0][0].shape[:2])
    _s = 0  # progress bar
    _itrs = lnE.shape[0]*lnE.shape[1]*len(itp[0])
    for x in range(lnE.shape[0]):
        for y in range(lnE.shape[1]):
            lnEiu = 0
            lnEid = 0.0000001  # To make sure not divide by 0
            for j in range(len(itp[0])):
                Zij = itp[0][j][x, y, channel]  # Pixel value
                lnEiu += w(Zij)*(g[Zij] - np.log(itp[1][j]))
                lnEid += w(Zij)
                _s += 1
                if (_s%100000 == 0):
                    print(("\tProgress %.8f" % (100.*_s / _itrs)) +'%' , end='\r')
            lnE[x, y] = lnEiu / lnEid
    return lnE


def intensitySample(images):
    z_min, z_max = 0, 255
    num_intensities = z_max - z_min + 1
    num_images = len(images)
    intensity_values = np.zeros((num_intensities, num_images), dtype=np.uint8)

    # Find the middle image to use as the source for pixel intensity locations
    mid_img = images[num_images // 2]

    for i in range(z_min, z_max + 1):
        rows, cols = np.where(mid_img == i)
        if len(rows) != 0:
            idx = random.randrange(len(rows))
            for j in range(num_images):
                intensity_values[i, j] = images[j][rows[idx], cols[idx]]
    return intensity_values


def reconstructRadiance(images, response_curve, log_exposure_times):
    img_shape = images[0].shape
    img_rad_map = np.zeros(img_shape, dtype=np.float64)

    num_images = len(images)
    for i in range(img_shape[0]):
        print(i)
        for j in range(img_shape[1]):
            g = np.array([response_curve[images[k][i, j]] for k in range(num_images)])
            w_val = np.array([w(images[k][i, j]) for k in range(num_images)])
            SumW = np.sum(w_val)
            if SumW > 0:
                img_rad_map[i, j] = np.sum(w_val * (g - log_exposure_times) / SumW)
            else:
                img_rad_map[i, j] = g[num_images // 2] - log_exposure_times[num_images // 2]
    return img_rad_map


def hdr(images, ex, l):
    num_channels = images[0].shape[2]
    hdr_img = np.zeros((images[0].shape[0],
                        images[0].shape[1], images[0].shape[2]))

    for channel in range(num_channels):
        channel_stack = [img[:, :, channel] for img in images]

        Z = intensitySample(channel_stack)

        g, lE = objective(Z, ex, l)

        hdr_img[:, :, channel] = reconstructRadiance(channel_stack, g, ex)

    # Return the correct format
    hdr_img = np.exp(hdr_img)
    hdr_img[hdr_img > 255] = 255
    hdr_img[hdr_img < 0] = 0

    return hdr_img.astype(np.uint8)

if __name__ == "__main__":
    _x = np.arange(0,256, 1, int)
    imgs = loadImages("../hdr-bilder/Ocean")
    print("Creating sample set:")
    z, t = getZT(imgs, 2500)
    # eh.showImageData(z, "hm")
    smoothing = 20
    print("Get R response curve")
    gR, lE = objective(z[:, :, 0], t, smoothing)
    plt.plot(_x, gR)
    plt.show()
    lnE = reconstruct(imgs, gR, 0)
    hdrImage = np.exp(lnE)
    print(np.max(hdrImage))
    plt.imshow(lnE)
    plt.show()
    plt.imshow(hdrImage)
    plt.show()