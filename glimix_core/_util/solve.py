import warnings

from numpy import (
    abs as npy_abs,
    absolute,
    arctan2,
    array,
    atleast_1d,
    cos,
    divide,
    errstate,
    full_like,
    isfinite,
    logical_and,
    logical_not,
    maximum,
    minimum,
    nan_to_num,
    sign,
    sin,
    sqrt,
    zeros,
)
from numpy.linalg import LinAlgError, pinv


def nice_inv(A):
    """
    Nice inverse.
    """
    from numpy_sugar.linalg import sum2diag

    return pinv(sum2diag(A, 1e-12))


def rsolve(A, y):
    """
    Robust solve Ax=y.
    """
    from numpy_sugar.linalg import rsolve as _rsolve

    try:
        beta = _rsolve(A, y)
    except LinAlgError:
        msg = "Could not converge to solve Ax=y."
        msg += " Setting x to zero."
        warnings.warn(msg, RuntimeWarning)
        beta = zeros(A.shape[0])

    return beta


def heigvals(a, b, d):
    a = float(a)
    b = atleast_1d(b)
    d = atleast_1d(d)
    T = a + d
    D = a * d - b * b

    t0 = T / 2
    t1 = sqrt(maximum(T * T / 4 - D, 0))
    eig0 = t0 + t1
    eig1 = t0 - t1

    eig0 = absolute(eig0)
    eig1 = absolute(eig1)

    with errstate(invalid="ignore", divide="ignore"):
        return nan_to_num(maximum(eig0, eig1) / minimum(eig0, eig1))


def _hinv_svd(a, b, d):
    rcond = 1e-15
    U, S, VT = hsvd(a, b, d)

    maxi = maximum(absolute(S[0]), absolute(S[1]))
    cutoff = rcond * maxi

    large = S[0] > cutoff
    S[0] = divide(1, S[0], where=large, out=S[0])
    S[0][~large] = 0

    large = S[1] > cutoff
    S[1] = divide(1, S[1], where=large, out=S[1])
    S[1][~large] = 0

    SiVT = [[VT[0][0] * S[0], VT[0][1] * S[0]], [VT[1][0] * S[1], VT[1][1] * S[1]]]
    Ai = [
        [
            U[0][0] * SiVT[0][0] + U[0][1] * SiVT[1][0],
            U[0][0] * SiVT[0][1] + U[0][1] * SiVT[1][1],
        ],
        [
            U[1][0] * SiVT[0][0] + U[1][1] * SiVT[1][0],
            U[1][0] * SiVT[0][1] + U[1][1] * SiVT[1][1],
        ],
    ]
    ai = Ai[0][0]
    bi = Ai[0][1]
    di = Ai[1][1]

    return ai, bi, di


def hinv(a, b, d):
    rcond = 1e-7

    b = atleast_1d(b)
    d = atleast_1d(d)
    a = float(a)
    cond = heigvals(a, b, d)
    norm = a * d - b * b
    with errstate(invalid="ignore", divide="ignore"):
        ai = d / norm
        bi = -b / norm
        di = a / norm

    nok = cond > 1 / rcond
    ai[nok], bi[nok], di[nok] = _hinv_svd(a, b[nok], d[nok])
    return ai, bi, di


def _hinv(A00, A01, A11):
    from numpy_sugar import is_all_finite

    rcond = 1e-15
    b = atleast_1d(A01)
    d = atleast_1d(A11)
    a = full_like(d, A00)
    m = maximum(maximum(npy_abs(b), npy_abs(d)), abs(a))

    a /= m
    b = b / m
    c = b
    d = d / m

    bc = b * c
    ad = a * d
    with errstate(invalid="ignore", divide="ignore"):
        ai = a / (a * a - nan_to_num((bc * a) / d))
        bi = b / (b * b - nan_to_num(ad))
        di = d / (d * d - nan_to_num((bc * d) / a))

    ai /= m
    bi /= m
    di /= m

    ok = is_all_finite(ai) and is_all_finite(bi) and is_all_finite(di)
    if not ok:
        ok = logical_and.reduce([isfinite(ai), isfinite(bi), isfinite(di)])
        nok = logical_not(ok)
        U, S, VT = hsvd(a[nok], b[nok], d[nok])

        maxi = maximum(npy_abs(S[0]), npy_abs(S[1]))
        cutoff = rcond * maxi

        large = S[0] > cutoff
        S[0] = divide(1, S[0], where=large, out=S[0])
        S[0][~large] = 0

        large = S[1] > cutoff
        S[1] = divide(1, S[1], where=large, out=S[1])
        S[1][~large] = 0

        SiVT = [[VT[0][0] * S[0], VT[0][1] * S[0]], [VT[1][0] * S[1], VT[1][1] * S[1]]]
        Ai = [
            [
                U[0][0] * SiVT[0][0] + U[0][1] * SiVT[1][0],
                U[0][0] * SiVT[0][1] + U[0][1] * SiVT[1][1],
            ],
            [
                U[1][0] * SiVT[0][0] + U[1][1] * SiVT[1][0],
                U[1][0] * SiVT[0][1] + U[1][1] * SiVT[1][1],
            ],
        ]
        ai[nok] = Ai[0][0] / m
        bi[nok] = Ai[0][1] / m
        di[nok] = Ai[1][1] / m

    return ai, bi, di


def hsvd(a, b, d):
    a = atleast_1d(a)
    b = atleast_1d(b)
    d = atleast_1d(d)

    aa = a * a
    bb = b * b
    dd = d * d
    ab = a * b
    bd = b * d

    e = aa - dd
    s1 = aa + 2 * bb + dd
    s2 = sqrt(e ** 2 + 4 * (ab + bd) ** 2)

    t = 2 * ab + 2 * bd
    theta = arctan2(t, e) / 2
    psi = arctan2(t, aa - dd) / 2

    Ct = cos(theta)
    St = sin(theta)
    Cp = cos(psi)
    Sp = sin(psi)

    s11 = (a * Ct + b * St) * Cp + (b * Ct + d * St) * Sp
    s22 = (a * St - b * Ct) * Sp + (-b * St + d * Ct) * Cp

    U = [[Ct, -St], [St, Ct]]
    S = [sqrt((s1 + s2) / 2), sqrt(maximum((s1 - s2) / 2, 0.0))]

    VT = [[sign(s11) * Cp, sign(s11) * Sp], [-sign(s22) * Sp, sign(s22) * Cp]]

    # U S V.T
    return U, S, VT


def hsolve(A00, A01, A11, y0, y1):
    """
    Solver for the linear equations of two variables and equations only.

    It uses Householder reductions to solve A𝐱 = 𝐲 in a robust manner.

    Parameters
    ----------
    A : array_like
        Coefficient matrix.
    y : array_like
        Ordinate values.

    Returns
    -------
    ndarray
        Solution 𝐱.
    """
    from numpy_sugar import epsilon

    n = _norm(A00, A01)
    u0 = A00 - n
    u1 = A01
    nu = _norm(u0, u1)

    with errstate(invalid="ignore", divide="ignore"):
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

    with errstate(invalid="ignore", divide="ignore"):
        v0 = nan_to_num(u0 / nu)
        v1 = nan_to_num(u1 / nu)

    E00 = 1 - 2 * v0 * v0
    E01 = 0 - 2 * v0 * v1
    E11 = 1 - 2 * v1 * v1

    F00 = E00 * D00 + E01 * D01
    F01 = E01 * D11
    F11 = E11 * D11

    F11 = (npy_abs(F11) > epsilon.small) * F11

    with errstate(divide="ignore", invalid="ignore"):
        Fi00 = nan_to_num(F00 / F00 / F00)
        Fi11 = nan_to_num(F11 / F11 / F11)
        Fi10 = nan_to_num(-(F01 / F00) * Fi11)

    c0 = Fi00 * b0
    c1 = Fi10 * b0 + Fi11 * b1

    x0 = E00 * c0 + E01 * c1
    x1 = E01 * c0 + E11 * c1

    return array([x0, x1])


def _norm(x0, x1):
    m = maximum(npy_abs(x0), npy_abs(x1))
    with errstate(invalid="ignore"):
        a = (x0 / m) * (x0 / m)
        b = (x1 / m) * (x1 / m)
        return nan_to_num(m * sqrt(a + b))
