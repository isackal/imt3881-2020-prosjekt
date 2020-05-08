import numpy as np
import random


Z_MIN = 0
Z_MAX = 255


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


def objective(Z, B, l):
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

    A = np.zeros((Z.shape[0] * Z.shape[1] + n - 1, Z.shape[0] + n))
    b = np.zeros((A.shape[0], 1))

    # Include the data-fitting equations

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            w_ij = w(Z[i, j] + 1)
            A[k, Z[i, j] + 1] = w_ij
            A[k, n + i] = -w_ij
            b[k, 0] = w_ij * B[j]
            k += 1

    # Fix the curve by setting its middle value to 0

    A[k, 129] = 1
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

    g = x[0:n+1]
    lE = x[n:]

    return g, lE


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
