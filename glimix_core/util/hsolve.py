from numpy import array, sqrt, nan_to_num, errstate, maximum
from numpy import abs as npy_abs
from numpy_sugar import epsilon


def _norm(x0, x1):
    m = maximum(npy_abs(x0), npy_abs(x1))
    with errstate(invalid='ignore'):
        a = (x0 / m) * (x0 / m)
        b = (x1 / m) * (x1 / m)
        return nan_to_num(m * sqrt(a + b))


def hsolve(A00, A01, A11, y0, y1):
    r"""Solver for the linear equations of two variables and equations only.

    It uses Householder reductions to solve ``Ax = y`` in a robust manner.

    Parameters
    ----------
    A : array_like
        Coefficient matrix.
    y : array_like
        Ordinate values.

    Returns
    -------
    :class:`numpy.ndarray`  Solution ``x``.
    """

    n = _norm(A00, A01)
    u0 = A00 - n
    u1 = A01
    nu = _norm(u0, u1)

    with errstate(invalid='ignore', divide='ignore'):
        v0 = nan_to_num(u0 / nu)
        v1 = nan_to_num(u1 / nu)

    B00 = 1 - 2 * v0 * v0
    B01 = 0 - 2 * v0 * v1
    B11 = 1 - 2 * v1 * v1

    D00 = B00 * A00 + B01 * A01
    D01 = B00 * A01 + B01 * A11
    D11 = B01 * A01 + B11 * A11

    b0 = y0 - 2 * y0 * v0 * v0 - 2 * y1 * v0 * v1
    b1 = y1 - 2 * y0 * v1 * v0 - 2 * y1 * v1 * v1

    n = _norm(D00, D01)
    u0 = D00 - n
    u1 = D01
    nu = _norm(u0, u1)

    with errstate(invalid='ignore', divide='ignore'):
        v0 = nan_to_num(u0 / nu)
        v1 = nan_to_num(u1 / nu)

    E00 = 1 - 2 * v0 * v0
    E01 = 0 - 2 * v0 * v1
    E11 = 1 - 2 * v1 * v1

    F00 = E00 * D00 + E01 * D01
    F01 = E01 * D11
    F11 = E11 * D11

    F11 = (npy_abs(F11) > epsilon.small) * F11

    with errstate(divide='ignore', invalid='ignore'):
        Fi00 = nan_to_num(F00 / F00 / F00)
        Fi11 = nan_to_num(F11 / F11 / F11)
        Fi10 = nan_to_num(-(F01 / F00) * Fi11)

    c0 = Fi00 * b0
    c1 = Fi10 * b0 + Fi11 * b1

    x0 = E00 * c0 + E01 * c1
    x1 = E01 * c0 + E11 * c1

    return array([x0, x1])
