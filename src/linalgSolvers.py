import numpy as np
from scipy.sparse import spdiags
import scipy.sparse.linalg as lnalg
import imageio as im
import matplotlib.pyplot as plt
import threading


def explisittPart(u, h, a):
    # Kan senere erstattes med Aslaks diffuse
    u1 = -4*u
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
    solver="b"
):
    """
    Used to solve A(s, n)x = b, where s and n are used to
    create a sparse matrix where s is the coefficient on the main
    diagonal (aka self diagonal), and n is the coefficients on the
    "neighbour" diagonals

    Parameters
    ----------

    s   :   <float>, <ndarray>
            main diagonal coefficient(s)
    n   :   <float>, <ndarray>
            neighbour diagonals coefficient(s)
    b   :   <ndarray>
            rightside part of the equation, also is an image
    method  :   <char>
                n   =>  neuman rand condition
    solver  :   <char>
                defualt =>  exact solution
                b       =>  bicstab solution


    Returns
    -------

    np.reshape(ret, (y, x)) :   <ndarray>
                                result x of the equation as image
    """
    x = np.size(b, 1)
    y = np.size(b, 0)
    z = x*y
    b = np.ravel(b)  # Vectorize the image
    numbered = np.arange(1, z+1, 1, np.int)
    maskArr = numbered % x != 0
    _1 = np.ones(z)
    mainDiag = _1*s
    arrD2 = np.copy(_1)
    arrD1 = arrD2*maskArr
    arrU1 = np.concatenate(([0], arrD1[:-1]))
    arrU2 = np.concatenate((np.zeros(x), arrD2[:-x]))

    if method == 'n':
        """
        With the neumann rand condition, we add the following
        to the main diagonal:
        2n n n ... n n 2n   n 0 ... 0 n n 0 ... 0 n ... 2n n n ... n n 2n
        where n = -alpha
        """
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
        ret = lnalg.bicgstab(A, b, x0=b, atol=0.005)[0]
    else:
        ret = lnalg.spsolve(A, b)
    return np.reshape(ret, (y, x))


def solveAsnST(
    s,
    n,
    b,
    lck,
    resBuffer,
    channel,
    method,
    solver
):
    """
    Used to solve A(s, n)x = b for Single Thread, where s and n are used to
    create a sparse matrix where s is the coefficient on the main
    diagonal (aka self diagonal), and n is the coefficients on the
    "neighbour" diagonals

    Parameters
    ----------

    s   :   <float>, <ndarray>
            main diagonal coefficient(s)

    n   :   <float>, <ndarray>
            neighbour diagonals coefficient(s)

    b   :   <ndarray>
            rightside part of the equation, also is an image

    lck :   <Lock>
            mutex/Lock to the resource resBuffer

    resBuffer   :   <ndarray>
                    shared resource between threads

    channel     :   <int>
                    position where the result is stored in the resBuffer

    method      :   <char>
                    n   =>  neuman rand condition

    solver      :   <char>
                    defualt =>  exact solution
                    b       =>  bicstab solution
    """
    res = solveAsn(s, n, b, method, solver)
    lck.acquire()
    resBuffer[:, :, channel] = res
    lck.release()


def solveAsnRgb(
    s,
    n,
    b,
    method='n',
    solver="b"
):
    """
    Used to solve A(s, n)x = b for multichannel picture,
    where s and n are used to create a sparse matrix
    where s is the coefficient on the
    main diagonal (aka self diagonal), and n is the coefficients on the
    "neighbour" diagonals

    Uses one thread for each channel.

    Parameters
    ----------

    s   :   <float>, <ndarray>
            main diagonal coefficient(s)
    n   :   <float>, <ndarray>
            neighbour diagonals coefficient(s)
    b   :   <ndarray>
            rightside part of the equation, also is an image
    method  :   <char>
                n   =>  neuman rand condition
    solver  :   <char>
                defualt =>  exact solution
                b       =>  bicstab solution


    Returns
    -------

    np.reshape(ret, (y, x)) :   <ndarray>
                                result x of the equation as image
    """
    if len(b.shape) == 2:
        return solveAsn(s, n, b, method, solver)
    rsrc = np.zeros(b.shape)
    lock = threading.Lock()
    threads = []
    for i in range(b.shape[2]):
        t = threading.Thread(
            target=solveAsnST,
            args=(
                s,
                n,
                b[:, :, i],
                lock,
                rsrc,
                i,
                method,
                solver
            )
        )
        t.start()
        threads.append(t)

    for i in range(b.shape[2]):
        threads[i].join()

    return rsrc


def direkteGlatting(u, l, dx=1):
    return solveAsnRgb(
        4+l*dx*dx,
        -1,
        u*l*dx*dx,
        solver="b"
    )


def poissonImplisitt(
    u,
    h=0,
    alpha=0.2  # Can be very high
):
    """
    Iterative solution of the poisson equation with Implisitt solution.
    Does one iteration.

    Parameters
    ----------

    u       :   <ndarray>
                image

    h       :   <float>, <ndarray>
                h part in the poisson equation

    alpha   :   <float>
                alpha = dt/dx^2, alpha part of the poisson equation

    Returns
    -------

    <ndarray>
    image result
    """
    return solveAsnRgb(
        4*alpha + 1,
        -alpha,
        u - h*alpha
    )


def poissonCrank(
    u,
    h=0,
    alpha=0.4  # Best if alpha < 0.5
):
    """
    Iterative solution of the poisson equation with Crank Nicolson.
    Does one iteration.

    Parameters
    ----------

    u       :   <ndarray>
                image

    h       :   <float>, <ndarray>
                h part in the poisson equation

    alpha   :   <float>
                alpha = dt/dx^2, alpha part of the poisson equation
                should be less than 0,5. 0,5 may be slightly unstable.

    Returns
    -------

    <ndarray>
    image result
    """
    return solveAsnRgb(
        2 * alpha + 1,
        -alpha / 2,
        explisittPart(u, h, alpha/2)
    )