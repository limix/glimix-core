from __future__ import division

from numpy import all as npall
from numpy import sum as npsum
from numpy import (
    asarray, clip, dot, empty, errstate, full, inf, isfinite, log, nan_to_num,
    zeros
)
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon
from numpy_sugar.linalg import rsolve, solve
from tqdm import tqdm

from ..io import wprint
from ..util import log2pi


class FastScanner(object):
    r"""Approximated fast inference over several covariates.


    Parameters
    ----------
    y : array_like
        Real-valued outcome.
    X : array_like
        Matrix of covariates.
    QS : tuple
        Economic eigen decomposition ``((Q0, Q1), S0)``.
    v : float
        Variance due to iid effect.

    Attributes
    ----------
    _yTQDi : tuple
        :math:`\mathbf y^{\intercal}\mathrm Q_i\mathrm D_i^{-1}`

    _yTBy : tuple
        :math:`\mathbf y^{\intercal}\mathrm B_i\mathbf y`

    _yTBX : tuple
        :math:`\mathbf y^{\intercal}\mathrm B_i\mathrm X`

    _XTQDi : tuple
        :math:`\mathrm X^{\intercal}\mathrm Q_i\mathrm D_i^{-1}`.

    _ETBE : tuple
        :math:`\mathrm E_i^{\intercal}\mathrm B_i^{-1}\mathrm E_i`.

    _XTBX : tuple
        :math:`\mathrm X^{\intercal}\mathrm B_i\mathrm X`.

    _XTBM : tuple
        :math:`\mathrm X^{\intercal}\mathrm B_i\mathrm M_j`.

    _MTBX : tuple
        :math:`\mathrm M_j^{\intercal}\mathrm B_i\mathrm X`.

    _MTBM : tuple
        :math:`\mathrm M_j^{\intercal}\mathrm B_i\mathrm M_j`.
    """

    def __init__(self, y, X, QS, v):

        self._scale = None
        self._QS = QS
        self._D = [QS[1] + v, v]

        yTQ = [dot(y.T, Q) for Q in QS[0]]
        XTQ = [dot(X.T, Q) for Q in QS[0]]

        self._yTQDi = [l / r for (l, r) in zip(yTQ, self._D)]

        self._yTBy = [(i**2 / j).sum() for (i, j) in zip(yTQ, self._D)]

        self._yTBX = [dot(i, j.T) for (i, j) in zip(self._yTQDi, XTQ)]

        self._XTQDi = [i / j for (i, j) in zip(XTQ, self._D)]

        self._ETBE = ETBE(self._XTQDi, XTQ)

    @property
    def _nsamples(self):
        return self._QS[0][0].shape[0]

    def _static_lml(self):
        n = self._nsamples
        p = len(self._D[0])
        static_lml = -n * log2pi - n
        static_lml -= npsum(log(self._D[0]))
        static_lml -= (n - p) * log(self._D[1])
        return static_lml

    def _fast_scan_chunk(self, markers):
        markers = asarray(markers, float)

        if not markers.ndim == 2:
            raise ValueError("`markers` array must be bidimensional.")

        if not npall(isfinite(markers)):
            raise ValueError("One or more variants have non-finite value.")

        MTQ = [dot(markers.T, Q) for Q in self._QS[0]]

        yTBM = [dot(i, j.T) for (i, j) in zip(self._yTQDi, MTQ)]
        XTBM = [dot(i, j.T) for (i, j) in zip(self._XTQDi, MTQ)]
        MTBM = [npsum((i / j) * i, axis=1) for (i, j) in zip(MTQ, self._D)]

        nsamples = markers.shape[0]
        nmarkers = markers.shape[1]

        lmls = full(nmarkers, self._static_lml())
        effect_sizes = empty(nmarkers)

        if self._ETBE.ncovariates == 1:
            return self._fast_scan_chunk_1covariate_loop(
                lmls, effect_sizes, yTBM, XTBM, MTBM, nsamples)
        else:
            return self._fast_scan_chunk_multicovariate_loop(
                lmls, effect_sizes, yTBM, XTBM, MTBM, nsamples, nmarkers)

    def _fast_scan_chunk_multicovariate_loop(self, lmls, effect_sizes, yTBM,
                                             XTBM, MTBM, nsamples, nmarkers):
        yTBE = [empty(len(self._yTBX[0]) + 1), empty(len(self._yTBX[1]) + 1)]

        yTBE[0][:-1] = self._yTBX[0]
        yTBE[1][:-1] = self._yTBX[1]

        for i in range(nmarkers):

            yTBE[0][-1] = yTBM[0][i]
            yTBE[1][-1] = yTBM[1][i]

            self._ETBE.XTBM(0)[:] = XTBM[0][:, i]
            self._ETBE.XTBM(1)[:] = XTBM[1][:, i]

            self._ETBE.MTBX(0)[:] = self._ETBE.XTBM(0)[:]
            self._ETBE.MTBX(1)[:] = self._ETBE.XTBM(1)[:]

            self._ETBE.MTBM(0)[:] = MTBM[0][i]
            self._ETBE.MTBM(1)[:] = MTBM[1][i]

            beta = _try_solve(self._ETBE.value[1] - self._ETBE.value[0],
                              yTBE[1] - yTBE[0])

            effect_sizes[i] = beta[-1]

            if self._scale is None:
                # _compute_scale(nsamples, beta, self._yTBy,
                # self._yTBX, )
                # (nsamples, beta, yTBy, yTBX, yTBM, XTBX,
                #                    XTBM, MTBM)

                p0 = self._yTBy[0] - 2 * yTBE[0].dot(beta) + beta.dot(
                    self._ETBE.value[0]).dot(beta)
                p1 = self._yTBy[1] - 2 * yTBE[1].dot(beta) + beta.dot(
                    self._ETBE.value[1].dot(beta))

                scale = (p0 + p1) / nsamples
            else:
                scale = self._scale

            lmls[i] -= nsamples * log(max(scale, epsilon.super_tiny))

        lmls /= 2
        return lmls, effect_sizes

    def _fast_scan_chunk_1covariate_loop(self, lmls, effect_sizes, yTBM, XTBM,
                                         MTBM, nsamples):

        sC00 = self._ETBE.XTBX(1)[0, 0] - self._ETBE.XTBX(0)[0, 0]
        sC01 = XTBM[1][0, :] - XTBM[0][0, :]
        sC11 = MTBM[1] - MTBM[0]

        sb = self._yTBX[1][0] - self._yTBX[0][0]
        sbm = yTBM[1] - yTBM[0]

        with errstate(divide='ignore'):
            beta = _beta_1covariate(sb, sbm, sC00, sC01, sC11)

        beta = [nan_to_num(bet) for bet in beta]

        scale = zeros(len(lmls))

        if self._scale is None:
            for i in range(2):
                scale += self._yTBy[i] - 2 * (
                    self._yTBX[i][0] * beta[0] + yTBM[i] * beta[1])
                scale += beta[0] * (self._ETBE.XTBX(i)[0, 0] * beta[0] +
                                    XTBM[i][0, :] * beta[1])
                scale += beta[1] * (
                    XTBM[i][0, :] * beta[0] + MTBM[i] * beta[1])
            scale /= nsamples
        else:
            scale = self._scale

        lmls -= nsamples * log(clip(scale, epsilon.super_tiny, inf))
        lmls /= 2

        effect_sizes = beta[1]

        return lmls, effect_sizes

    def fast_scan(self, markers, verbose=True):
        r"""LML and fixed-effect sizes of each marker.

        If the scaling factor ``s`` is not set by the user via method
        :func:`set_scale`, its optimal value will be found and
        used in the calculation.

        Parameters
        ----------
        markers : array_like
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

        if not (markers.ndim == 2):
            raise ValueError("`markers` array must be bidimensional.")
        p = markers.shape[1]

        lmls = empty(p)
        effect_sizes = empty(p)

        if verbose:
            nchunks = min(p, 30)
        else:
            nchunks = min(p, 1)

        chunk_size = (p + nchunks - 1) // nchunks

        for i in tqdm(range(nchunks), desc="Scanning", disable=not verbose):
            start = i * chunk_size
            stop = min(start + chunk_size, markers.shape[1])

            l, e = self._fast_scan_chunk(markers[:, start:stop])

            lmls[start:stop] = l
            effect_sizes[start:stop] = e

        return lmls, effect_sizes

    def null_lml(self):
        r"""Log of the marginal likelihood.

        .. math::

            - \frac{n}{2}\log{2\pi}
                - \frac{1}{2} n \log{s}
                - \frac{1}{2} \log{\left|\mathrm D\right|}
                    - \frac{1}{2}
                \left(\mathrm Q^{\intercal}\mathbf y -
              \mathrm Q^{\intercal}\mathrm X\boldsymbol\beta\right)^{\intercal}
                    s^{-1}\mathrm D^{-1}
                \left(\mathrm Q^{\intercal}\mathbf y -
                \mathrm Q^{\intercal}\mathrm X\boldsymbol\beta\right)
        """
        n = self._QS[0][0].shape[0]
        p = len(self._D[0])
        static_lml = -n * log2pi - n
        static_lml -= npsum(log(self._D[0]))
        static_lml -= (n - p) * log(self._D[1])
        return static_lml

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

        for i in range(2):
            self._data[i][:-1, :-1] = dot(XTQDi[i], XTQ[i].T)

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

    # def set_markers(self, markers):
    #     markers
    #     MTQ = [dot(markers.T, Q) for Q in self._QS[0]]
    #
    #     yTBM = [dot(i, j.T) for (i, j) in zip(self._yTQDi, MTQ)]
    #     XTBM = [dot(i, j.T) for (i, j) in zip(self._XTQDi, MTQ)]
    #     MTBM = [npsum((i / j) * i, axis=1) for (i, j) in zip(MTQ, self._D)]


def _beta_1covariate(sb, sbm, sC00, sC01, sC11):
    d0 = sb / sC00
    d1 = sb / sC01

    d3 = sbm / sC01
    d4 = sbm / sC11

    d5 = sC00 / sC01
    d6 = sC11 / sC01

    return [(d1 - d4) / (d5 - 1 / d6), (-d0 + d3) / (d6 - 1 / d5)]


def _compute_scale(nsamples, beta, yTBy, yTBX, yTBM, XTBX, XTBM, MTBM):

    scale = npsum(yTBy[i] for i in range(2))
    scale -= npsum(2 * yTBX[i] * beta[0] for i in range(2))
    scale -= npsum(2 * yTBM[i] * beta[1] for i in range(2))

    scale += npsum(beta[0] * XTBX[i] * beta[0] for i in range(2))
    scale += npsum(beta[0] * XTBM[i] * beta[1] for i in range(2))
    scale += npsum(beta[1] * XTBM[i] * beta[0] for i in range(2))
    scale += npsum(beta[1] * MTBM[i] * beta[1] for i in range(2))

    return scale / nsamples


def _try_solve(A, y):

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
