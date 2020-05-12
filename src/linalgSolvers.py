import numpy as np
from scipy.sparse import spdiags, csr_matrix, csc_matrix
import scipy.sparse.linalg as lnalg
import imageio as im
import matplotlib.pyplot as plt
from time import time

def explisittPart(u, h, a):
    # Kan senere erstattes med Aslaks diffuse
    u1 = -4*u.astype(np.int)
    u1[0:-1, :] += u[1:, :]
    u1[1:, :] += u[0:-1, :]
    u1[:, 0:-1] += u[:, 1:]
    u1[:, 1:] += u[:, 0:-1]
    # Neumann condition
    u1[0, :] += u[0, :]
    u1[:, 0] += u[:, 0]
    u1[-1, :] += u[-1, :]
    u1[:, -1] += u[:, -1]
    # return (u1 * a + h * a + u)
    return u + (u1*a) - (h*a)

def solveAsn(
    s,
    n,
    b,
    method='n',
    solver="b"):
    """
    Used to solve A(s, n)x = b
    """
    x = np.size(b, 1)
    y = np.size(b, 0)
    z = x*y
    b = np.ravel(b)  # Vectorize the image
    numbered = np.arange(1, z+1, 1, np.int)
    maskArr = numbered % x != 0
    _1 = np.ones(z)
    mainDiag =  _1*s
    arrD2 = np.copy(_1)
    arrD1 = arrD2*maskArr
    arrU1 = np.concatenate(([0], arrD1[:-1]))
    arrU2 = np.concatenate((np.zeros(x), arrD2[:-x]))

    if method == 'n':
        # With the neumann rand condition, we add the following to the main diagonal:
        # 2n n n ... n n 2n   n 0 ... 0 n n 0 ... 0 n ... 2n n n ... n n 2n
        # where n = -alpha
        _mask1 = ((numbered <= x) + (numbered > z-x))  # numbered starts from 1
        _mask2 = ((numbered-1) % x == 0) + (numbered % x == 0)
        _ad = (_1*_mask1 + _1*_mask2)
        _nmnArr = _ad*n
        mainDiag += _nmnArr

    diags = np.array([
        0, -1, 1, -x, x
    ])

    data = np.array([
        mainDiag,
        arrD1*n,
        arrU1*n,
        arrD2*n,
        arrU2*n
    ])
    A = spdiags(data, diags, z, z).tocsr()
    ret = None
    if solver == "b":
        ret = lnalg.bicgstab(A, b, x0=b, atol=0.5)[0]
    else:
        ret = lnalg.spsolve(A, b)
    return np.reshape(ret, (y, x))


def direkteGlatting(u, l, dx=1):
    return solveAsn(
        4+l*dx*dx,
        -1,
        u*l*dx*dx,
        solver="b"
    )


def diffuseImplisitt(
    u,
    h=0,
    alpha=0.2,
    funcN=(lambda a: -a),
    funcS=(lambda a: 4 * a + 1),
    funcImg=(lambda _u, _h, a: _u - _h*a),
    method='n'):
    n = funcN(alpha)
    s = funcS(alpha)
    x = np.size(u, 1)
    y = np.size(u, 0)
    z = x*y
    b = np.ravel(funcImg(u, h, alpha))  # Vectorize the image

    numbered = np.arange(1, z+1, 1, np.int)
    maskArr = numbered % x != 0
    _1 = np.ones(z)
    mainDiag = _1*s
    arrD2 = _1*n
    arrD1 = arrD2*maskArr
    arrU1 = np.concatenate(([0], arrD1[:-1]))
    arrU2 = np.concatenate((np.zeros(x), arrD2[:-x]))

    if method == 'n':
        # With the neumann rand condition, we add the following to the main diagonal:
        # 2n n n ... n n 2n   n 0 ... 0 n n 0 ... 0 n ... 2n n n ... n n 2n
        # where n = -alpha
        _mask1 = ((numbered <= x) + (numbered > z-x))  # numbered starts from 1
        _mask2 = ((numbered-1) % x == 0) + (numbered % x == 0)
        _ad = (_1*_mask1 + _1*_mask2)
        _nmnArr = _ad * n
        mainDiag += _nmnArr

    diags = np.array([
        0, -1, 1, -x, x
    ])

    data = np.array([
        mainDiag,
        arrD1,
        arrU1,
        arrD2,
        arrU2
    ])
    A = spdiags(data, diags, z, z).tocsr()
    ret = lnalg.spsolve(A, b)
    return np.reshape(ret, (y, x))

def crankNicolson(
    u,
    __h=0,
    __alpha=10
):
    return diffuseImplisitt(
        u,
        h=__h,
        alpha=__alpha,
        funcN=(lambda a: -a/2),
        funcS=(lambda a: 2*a +1),
        funcImg=(lambda _u, _h, a: explisittPart(u, h, a/2)),
        method='n'
    )

if __name__ == "__main__":
    orig_im = im.imread("../testimages/raccoon.png").astype(float) / 255
    for i in range(3):
        orig_im[:, :, i] = direkteGlatting(orig_im[:, :, i], 0.01)
    plt.imshow(orig_im)
    plt.show()
