import numpy as np


def diffuse(u, alpha=0.24, h=0, D=1., rand='n', dr=0):
    """
    The general diffusion equation, used to
    diffuse a one-channel-float[0, 1]-image.
    """
    u1 = -4*u
    u1[0:-1, :] += u[1:, :]
    u1[1:, :] += u[0:-1, :]
    u1[:, 0:-1] += u[:, 1:]
    u1[:, 1:] += u[:, 0:-1]
    # Randbetingelser:
    if rand == 'n':
        # Neumann condition
        u1[0, :] += u[0, :]
        u1[:, 0] += u[:, 0]
        u1[-1, :] += u[-1, :]
        u1[:, -1] += u[:, -1]
    else:
        u1[0, :] += dr
        u1[:, 0] += dr
        u1[-1, :] += dr
        u1[:, -1] += dr
    return (u1 * D * alpha + h + u)


def gY(img):
    """
    Gradient Y component of an image.
    """
    img1 = np.zeros(img.shape)
    img1[1:-1, :] = (img[2:, :] - img[:-2, :])/2
    img1[0, :] = img[1, :]-img[0, :]
    img1[-1, :] = img[-1, :]-img[-2, :]
    return img1


def gX(img):
    """
    Gradient X component of an image.
    """
    img1 = np.zeros(img.shape)
    img1[:, 1:-1] = (img[:, 2:] - img[:, :-2])/2
    img1[:, 0] = img[:, 1]-img[:, 0]
    img1[:, -1] = img[:, -1]-img[:, -2]
    return img1


def D_Image(img, k):
    timg = np.zeros(img.shape)
    for i in range(timg.shape[2]):
        timg[:, :, i] = 1. / (1 + k * (gX(img)**2 + gY(img)**2))
    return timg


#TODO make a smarter way to diffuse image (diffuse arbitrary channels in one go)
def pre_diffuse(u, mask, methode, rand, alpha, itr, h, D):
    u1 = np.copy(u)
    if mask is None:
        pass
    else:
        mask = mask.astype(bool)
        maskCords = np.argwhere(mask)

        # Generate view of image
        top = np.amin(maskCords[:, 0])
        bottom = np.amax(maskCords[:, 0]) + 1
        left = np.amin(maskCords[:, 1])
        right = np.amax(maskCords[:, 1]) + 1

        view = u[top:bottom, left:right]
        new_view = u1[top:bottom, left:right]
        mask = mask[top:bottom, left:right]

        if methode == 'e': #Explicitt diffusion
            for i in range(itr):
                if len(u1.shape) > 2:
                    for j in range(3):
                        new_view[:, :, j] = explisitt(new_view[:, :, j], alpha, h, D)
                else:
                    new_view[:] = explisitt(new_view[:], alpha, h, D)
                
                new_view[~mask] = view[~mask]
                """
                TO BE TESTED:
                cloning
                grayscale
                Contrast
                """
        
        elif methode == 'i': #Implisitt diffusion
            pass
    
    return u1

def explisitt(u, alpha=0.24, h=0, D=1., dr=0):
    laplace = (
                u[1:-2, 2:-1] +
                u[1:-2, 0:-3] +
                u[2:-1, 1:-2] +
                u[0:-3, 1:-2] -
                4*u[1:-2, 1:-2]
            )
    u[1:-2, 1:-2] += laplace * alpha * D
    
    # Neumann boundary condition du/dt = 0
    u[0, :] = u[1, :]
    u[:, 0] = u[:, 1]
    u[-1, :] = u[-2, :]
    u[:, -1] = u[:, -2]

    return (u * 255 + h * 255) / 255