from __future__ import division

import numpy as np
import numpy.linalg
import scipy as sp
import scipy.stats
from numpy import all as npall
from numpy import sum as npsum
from numpy import (asarray, clip, dot, empty, errstate, full, inf, isfinite,
                   log, nan_to_num, zeros, pi)
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon
from numpy_sugar.linalg import rsolve, solve
from tqdm import tqdm

from ..util import wprint
from ..util import log2pi


class FastScanner(object):
    r"""Approximated fast inference over several covariates.

    Specifically, it computes the log of the marginal likelihood

    .. math::

        \log p(\mathbf y)_j = \log \mathcal N\big(~ \mathrm X\mathbf b_j^*
        + \mathbf{m}_j \alpha_j^*,~
        s_j^* (\mathrm K + v \mathrm I) ~\big),

    for optimal :math:`\mathbf b_j`, :math:`\alpha_j`, and :math:`s_j^*`
    values.
    Vector :math:`\mathbf{m}_j` is the candidate defined by the i-th column
    of matrix ``M`` provided to method
    :func:`glimix_core.lmm.FastScanner.fast_scan`.
    Variance :math:`v` is not optimised for performance reasons.
    The method assumes the user has provided a reasonable value for it.

    Notes
    -----
    The implementation requires further explanation as it is somehow obscure.
    Let :math:`\mathrm B_i=\mathrm Q_i\mathrm D_i^{-1}\mathrm Q_i^{\intercal}`
    for :math:`i \in \{0, 1\}` and
    :math:`\mathrm E_j = [\mathrm X;~ \mathbf m_j]`.
    The matrix resulted from
    :math:`\mathrm E_j^{\intercal}\mathrm B_i\mathrm E_j`
    is represented by the attribute ``_ETBE``, and
    four views of such a matrix are given by the attributes ``_XTBX``,
    ``_XTBM``, ``_MTBX``, and ``_MTBM``.
    Those views represent
    :math:`\mathrm X^{\intercal}\mathrm B_i\mathrm X`,
    :math:`\mathrm X^{\intercal}\mathrm B_i\mathbf m_j`,
    :math:`\mathbf m_j^{\intercal}\mathrm B_i\mathrm X`, and
    :math:`\mathbf m_j^{\intercal}\mathrm B_i\mathbf m_j`, respectively.

    Parameters
    ----------
    y : array_like
    Real-valued outcome.
    X : array_like
    Matrix of covariates.
    QS : tuple
    Economic eigendecomposition ``((Q0, Q1), S0)`` of ``K``.
    v : float
    Variance due to iid effect.
    """

    def __init__(self, y, X, QS, v):

        self._y = y
        self._X = X
        self._scale = None
        self._QS = QS
        self._D = [QS[1] + v, v]
        self._v = v

        yTQ = [dot(y.T, Q) for Q in QS[0]]
        XTQ = [dot(X.T, Q) for Q in QS[0]]

        self._yTQDi = [l / r for (l, r) in zip(yTQ, self._D) if np.min(r) > 0]

        self._yTBy = [(i**2 / j).sum() for (i, j) in zip(yTQ, self._D)
                      if np.min(j) > 0]

        self._yTBX = [dot(i, j.T) for (i, j) in zip(self._yTQDi, XTQ)]

        self._XTQDi = [i / j for (i, j) in zip(XTQ, self._D) if np.min(j) > 0]

        self._ETBE = ETBE(self._XTQDi, XTQ)

    @property
    def _nsamples(self):
        return self._QS[0][0].shape[0]

    def _static_lml(self):
        n = self._nsamples
        p = len(self._D[0])
        static_lml = -n * log2pi - n

        D0 = clip(self._D[0], epsilon.super_tiny, inf)
        static_lml -= npsum(log(D0))

        D1 = clip(self._D[1], epsilon.super_tiny, inf)
        static_lml -= (n - p) * log(D1)
        return static_lml

    def _fast_scan_chunk(self, M):
        M = asarray(M, float)

        if not M.ndim == 2:
            raise ValueError("`M` array must be bidimensional.")

        if not npall(isfinite(M)):
            raise ValueError("One or more variants have non-finite value.")

        MTQ = [dot(M.T, Q) for Q in self._QS[0]]

        yTBM = [dot(i, j.T) for (i, j) in zip(self._yTQDi, MTQ)]
        XTBM = [dot(i, j.T) for (i, j) in zip(self._XTQDi, MTQ)]
        MTBM = [
            npsum((i / j) * i, axis=1) for (i, j) in zip(MTQ, self._D)
            if np.min(j) > 0
        ]

        nsamples = M.shape[0]
        nmarkers = M.shape[1]

        lmls = full(nmarkers, self._static_lml())
        effect_sizes = empty(nmarkers)

        if self._ETBE.ncovariates == 1:
            return self._fast_scan_chunk_1covariate_loop(
                lmls, effect_sizes, yTBM, XTBM, MTBM, nsamples, M)
        else:
            return self._fast_scan_chunk_multicovariate_loop(
                lmls, effect_sizes, yTBM, XTBM, MTBM, nsamples, nmarkers, M)

    def _fast_scan_chunk_multicovariate_loop(
            self, lmls, effect_sizes, yTBM, XTBM, MTBM, nsamples, nmarkers, M):

        yTBy = self._yTBy
        ETBE = self._ETBE
        yTBX = self._yTBX

        yTBE = [empty(len(i) + 1) for i in yTBX]

        for i in range(len(yTBE)):
            yTBE[i][:-1] = yTBX[i]

        for i in range(nmarkers):

            for j in range(len(yTBE)):
                yTBE[j][-1] = yTBM[j][i]
                ETBE.XTBM(j)[:] = XTBM[j][:, i]
                ETBE.MTBX(j)[:] = ETBE.XTBM(j)[:]
                ETBE.MTBM(j)[:] = MTBM[j][i]

            A = sum(ETBE.value[j] for j in range(len(yTBE)))
            b = sum(yTBE[j] for j in range(len(yTBE)))

            # beta = _solve(ETBE.value[1] + ETBE.value[0], yTBE[1] + yTBE[0])
            beta = _solve(A, b)

            effect_sizes[i] = beta[-1]

            p = 0
            for j in range(len(yTBy)):
                p += yTBy[j] - 2 * dot(yTBE[j], beta)
                p += dot(dot(beta, ETBE.value[j]), beta)

            if self._scale is None:
                scale = p / nsamples
            else:
                scale = self._scale
                lmls[i] = lmls[i] + nsamples
                lmls[i] = lmls[i] - p / scale

            lmls[i] -= nsamples * log(max(scale, epsilon.super_tiny))

        lmls /= 2
        return lmls, effect_sizes

    def _fast_scan_chunk_1covariate_loop(self, lmls, effect_sizes, yTBM, XTBM,
                                         MTBM, nsamples, M):

        ETBE = self._ETBE
        sC00 = sum(ETBE.XTBX(i)[0, 0] for i in range(ETBE.size))
        # sC00 = ETBE.XTBX(1)[0, 0] + ETBE.XTBX(0)[0, 0]
        # sC01 = XTBM[1][0, :] + XTBM[0][0, :]
        sC01 = sum(XTBM[i][0, :] for i in range(ETBE.size))
        # sC11 = MTBM[1] + MTBM[0]
        sC11 = sum(MTBM[i] for i in range(ETBE.size))

        # sb = self._yTBX[1][0] + self._yTBX[0][0]
        sb = sum(self._yTBX[i][0] for i in range(ETBE.size))
        # sbm = yTBM[1] + yTBM[0]
        sbm = sum(yTBM[i] for i in range(ETBE.size))

        beta = _beta_1covariate(sb, sbm, sC00, sC01, sC11)

        beta = [nan_to_num(bet) for bet in beta]

        scale = zeros(len(lmls))

        if self._scale is None:
            for i in range(ETBE.size):
                scale += self._yTBy[i] - 2 * (
                    self._yTBX[i][0] * beta[0] + yTBM[i] * beta[1])
                scale += beta[0] * (self._ETBE.XTBX(i)[0, 0] * beta[0] +
                                    XTBM[i][0, :] * beta[1])
                scale += beta[1] * (
                    XTBM[i][0, :] * beta[0] + MTBM[i] * beta[1])
            scale /= nsamples
        else:
            scale[:] = self._scale
            lmls += nsamples
            bla = zeros(len(lmls))
            for i in range(ETBE.size):
                bla += self._yTBy[i] - 2 * (
                    self._yTBX[i][0] * beta[0] + yTBM[i] * beta[1])
                bla += beta[0] * (self._ETBE.XTBX(i)[0, 0] * beta[0] +
                                  XTBM[i][0, :] * beta[1])
                bla += beta[1] * (XTBM[i][0, :] * beta[0] + MTBM[i] * beta[1])

            lmls -= bla / scale

        lmls -= nsamples * log(clip(scale, epsilon.super_tiny, inf))
        lmls /= 2

        effect_sizes = beta[1]

        return lmls, effect_sizes

    def fast_scan(self, M, verbose=True):
        r"""LML and fixed-effect sizes of each marker.

        If the scaling factor ``s`` is not set by the user via method
        :func:`set_scale`, its optimal value will be found and
        used in the calculation.

        Parameters
        ----------
        M : array_like
        Matrix of fixed-effects across columns.
        verbose : bool, optional
        ``True`` for progress information; ``False`` otherwise.
        Defaults to ``True``.

        Returns
        -------
        array_like
        Log of the marginal likelihoods.
        array_like
        Fixed-effect sizes.
        """

        if not (M.ndim == 2):
            raise ValueError("`M` array must be bidimensional.")
        p = M.shape[1]

        lmls = empty(p)
        effect_sizes = empty(p)

        if verbose:
            nchunks = min(p, 30)
        else:
            nchunks = min(p, 1)

        chunk_size = (p + nchunks - 1) // nchunks

        for i in tqdm(range(nchunks), desc="Scanning", disable=not verbose):
            start = i * chunk_size
            stop = min(start + chunk_size, M.shape[1])

            l, e = self._fast_scan_chunk(M[:, start:stop])

            lmls[start:stop] = l
            effect_sizes[start:stop] = e

        return lmls, effect_sizes

    def null_lml(self):

        # lml = self._static_lml()
        # beta = _solve(ETBE.value[1] + ETBE.value[0], yTBE[1] + yTBE[0])
        # XTBX = self._ETBE.XTBX.value(0) + self._ETBE.XTBX.value(1)
        # yTBX = self._yTBX[0] + self._yTBX[1]
        # beta = _solve(XTBX, yTBX)

        y = self._y
        X = self._X
        QS = self._QS
        n = len(y)

        D = np.zeros((n, n))
        Di = np.zeros((n, n))

        p = QS[0][0].shape[1] - QS[0][0].shape[0]
        d = np.concatenate((self._D[0], [self._D[1]] * p))
        D[np.diag_indices_from(D)] = d
        Di[np.diag_indices_from(D)] = 1 / d

        Q = np.concatenate(self._QS[0], axis=1)
        beta = _solve(
            X.T.dot(Q).dot(Di).dot(Q.T).dot(X),
            y.T.dot(Q).dot(Di).dot(Q.T).dot(X))

        if self._scale is None:
            scale = y.T.dot(Q).dot(Di).dot(Q.T).dot(y - X.dot(beta)) / n
            return (-n / 2) * (
                log(2 * pi) + log(scale) + 1) - log(D.diagonal()).sum() / 2

        scale = self._scale

        return (-n / 2) * (log(2 * pi) + log(scale)) - log(D.diagonal()).sum(
        ) / 2 + y.T.dot(Q).dot(Di).dot(Q.T).dot(X.dot(beta) - y) / (2 * scale)

    def set_scale(self, scale):
        r"""Set the scaling factor.

        Calling this method disables the automatic scale learning.

        Parameters
        ----------
        scale : float
        Scaling factor.
        """
        self._scale = scale

    def unset_scale(self):
        r"""Unset the scaling factor.

        If called, it enables the scale learning again.
        """
        self._scale = None


class ETBE(object):
    def __init__(self, XTQDi, XTQ):
        n = XTQDi[0].shape[0] + 1

        self._data = [empty((n, n)), empty((n, n))]

        self._data = []
        for i in range(len(XTQDi)):
            data = empty((n, n))
            data[:-1, :-1] = dot(XTQDi[i], XTQ[i].T)
            self._data.append(data)

    @property
    def size(self):
        return len(self._data)

    @property
    def ncovariates(self):
        return self.XTBX(0).shape[0]

    @property
    def value(self):
        return self._data

    def XTBX(self, i):
        return self._data[i][:-1, :-1]

    def XTBM(self, i):
        return self._data[i][:-1, -1]

    def MTBX(self, i):
        return self._data[i][-1, :-1]

    def MTBM(self, i):
        return self._data[i][-1:, -1:]


def _beta_1covariate(sb, sbm, sC00, sC01, sC11):

    with errstate(all='ignore'):
        d0 = nan_to_num(sb / sC00)
        d1 = nan_to_num(sb / sC01)

        d3 = nan_to_num(sbm / sC01)
        d4 = nan_to_num(sbm / sC11)

        d5 = nan_to_num(sC00 / sC01)
        d6 = nan_to_num(sC11 / sC01)

        a = (d1 - d4) / (d5 - 1 / d6)
        # a[abs(d5 - 1 / d6) <= epsilon.tiny] = 0

        b = (-d0 + d3) / (d6 - 1 / d5)
        # b[abs(d6 - 1 / d5) <= epsilon.tiny] = 0

    return nan_to_num([a, b])


def _compute_scale(nsamples, beta, yTBy, yTBX, yTBM, XTBX, XTBM, MTBM):

    scale = npsum(yTBy[i] for i in range(2))
    scale -= npsum(2 * yTBX[i] * beta[0] for i in range(2))
    scale -= npsum(2 * yTBM[i] * beta[1] for i in range(2))

    scale += npsum(beta[0] * XTBX[i] * beta[0] for i in range(2))
    scale += npsum(beta[0] * XTBM[i] * beta[1] for i in range(2))
    scale += npsum(beta[1] * XTBM[i] * beta[0] for i in range(2))
    scale += npsum(beta[1] * MTBM[i] * beta[1] for i in range(2))

    return scale / nsamples


def _solve(A, y):

    try:
        beta = solve(A, y)
    except LinAlgError:
        try:
            beta = rsolve(A, y)
        except LinAlgError:
            msg = "Could not converge to the optimal"
            msg += " effect-size of one of the candidates."
            msg += " Setting its effect-size to zero."
            wprint(msg)
            beta = zeros(A.shape[0])

    return beta


# def _lml_this(y, X, QS, _D, scale):
#     n = len(y)
#
#     D = np.zeros((n, n))
#     Di = np.zeros((n, n))
#
#     p = QS[0][0].shape[0] - QS[0][0].shape[1]
#     d = np.concatenate((_D[0], [_D[1]] * p))
#     D[np.diag_indices_from(D)] = d
#     Di[np.diag_indices_from(D)] = 1 / d
#
#     Q = np.concatenate(QS[0], axis=1)
#     beta = _solve(
#         X.T.dot(Q).dot(Di).dot(Q.T).dot(X), y.T.dot(Q).dot(Di).dot(Q.T).dot(X))
#
#     if scale is None:
#         scale = y.T.dot(Q).dot(Di).dot(Q.T).dot(y - X.dot(beta)) / n
#         scale = clip(scale, epsilon.super_tiny, inf)
#         lml = (-n / 2) * (
#             log(2 * pi) + log(scale) + 1) - log(D.diagonal()).sum() / 2
#     else:
#         lml = (-n / 2) * (log(2 * pi) + log(scale)) - log(D.diagonal()).sum(
#         ) / 2 + y.T.dot(Q).dot(Di).dot(Q.T).dot(X.dot(beta) - y) / (2 * scale)
#
#     return lml, beta[-1], beta
