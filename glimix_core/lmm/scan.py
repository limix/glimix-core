from __future__ import division

import logging

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

from ..util import log2pi


class FastScanner(object):
    r"""Approximated fast inference over several covariates.

    Let :math:`\mathrm X` be a samples-by-covariates matrix,
    :math:`\mathrm M` a samples-by-markers matrix, and
    :math:`\mathbf y` an array of outcome.
    A covariance :math:`\mathrm K` will be provided via its economic eigen
    decomposition ``((Q0, Q1), S0)`` and ``v`` will define the variance
    due to the identity in the formulae

    .. math::

        \mathbf y \sim \mathcal N\big(~ \mathrm X\boldsymbol\beta
        + \mathrm{M}_i \alpha_i,~
        s (\mathrm K + v \mathrm I) ~\big)

    Note that :math:`\alpha_i` is a scalar multiplying a column-matrix
    :math:`\mathrm{M}_i`.
    The variable :math:`s` is a scaling factor that, if not set, is jointly
    adjusted with :math:`\alpha_i` in order the maximise the marginal
    likelihood, ultimately providing the degree of association between the
    marker :math:`\mathrm{M}_i` with the outcome :math:`\mathbf y` via an
    p-value.

    For performance reasons, we make use of the identity

    .. math::

        (\mathrm K + v\mathrm I)^{-1} = \mathrm Q \left(
            \begin{array}{cc}
                \mathrm S_0 + v\mathrm I_0 & \mathbf 0\\
                \mathbf 0 & v\mathrm I_1
            \end{array}\right)^{-1}
            \mathrm Q^{\intercal}

    in the implementation.
    We can thus write the marginal likelihood as

    .. math::

        \mathcal N\left(\mathrm Q^{\intercal} \mathbf y ~|~
                   \mathrm Q^{\intercal} \mathbf m,~
                   s \left(
                       \begin{array}{cc}
                           \mathrm S_0 + v\mathrm I_0 & \mathbf 0\\
                           \mathbf 0 & v\mathrm I_1
                       \end{array}\right)\right).

    Expanding it gives us

    .. math::

        -\frac{n}{2} \log 2\pi - \frac{1}{2}n \log s
            - \frac{1}{2}\log|\mathrm D|\\
            - \frac{1}{2} (\mathrm Q^{\intercal}\mathbf y)^{\intercal} s^{-1}
            \mathrm D^{-1}(\mathrm Q^{\intercal} \mathbf y)\\
            + (\mathrm Q^{\intercal}\mathbf y)^{\intercal}
            s^{-1} \mathrm D^{-1}
            (\mathrm Q^{\intercal} \mathrm X\boldsymbol\beta)\\
            - \frac{1}{2} (\mathrm Q^{\intercal}
            \mathrm X\boldsymbol\beta)^{\intercal} s^{-1} \mathrm D^{-1}
            (\mathrm Q^{\intercal} \mathrm X\boldsymbol\beta),

    where

    .. math::

        \mathrm D = \left(
            \begin{array}{cc}
              \mathrm S_0 + v\mathrm I_0 & \mathbf 0\\
              \mathbf 0 & v\mathrm I_1
            \end{array}
            \right)

    Attributes
    ----------
    _yTQDiQTy : tuple
        Store the arrays

        .. math::

            (\mathrm Q_i^{\intercal} \mathbf y)^{\intercal}
            \mathrm D_i^{-1}
            (\mathrm Q_i^{\intercal} \mathbf y),

        for :math:`i = \{0, 1\}`.

    _yTQDiQTX : tuple
        Store the arrays

        .. math::

            (\mathrm Q_i^{\intercal} \mathbf y)^{\intercal}
            \mathrm D_i^{-1}
            (\mathrm Q_i^{\intercal} \mathrm X),

        for :math:`i = \{0, 1\}`.

    _yTQDi : tuple
        Store the arrays
        :math:`(\mathrm Q_i^{\intercal} \mathbf y)^{\intercal}\mathrm D_i^{-1}`
        , for :math:`i = \{0, 1\}`.

    _XTQDi : tuple
        Store the arrays
        :math:`(\mathrm Q_i^{\intercal} \mathrm X)^{\intercal}\mathrm D_i^{-1}`
        , for :math:`i = \{0, 1\}`.

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
    """

    def __init__(self, y, X, QS, v):

        self._scale = None
        self._QS = QS
        self._D = [QS[1] + v, v]

        yTQ = [dot(y.T, Q) for Q in QS[0]]

        XTQ = [dot(X.T, Q) for Q in QS[0]]

        self._yTQDi = [l / r for (l, r) in zip(yTQ, self._D)]

        self._yTQDiQTy = [(i**2 / j).sum() for (i, j) in zip(yTQ, self._D)]

        self._yTQDiQTX = [dot(i, j.T) for (i, j) in zip(self._yTQDi, XTQ)]

        self._XTQDi = [i / j for (i, j) in zip(XTQ, self._D)]

        nc = X.shape[1]

        self._C = [empty((nc + 1, nc + 1)), empty((nc + 1, nc + 1))]

        for i in range(2):
            self._C[i][:-1, :-1] = dot(self._XTQDi[i], XTQ[i].T)

        self._XTQDiQTX = [self._C[i][:-1, :-1] for i in range(2)]

    def _static_lml(self):
        n = self._QS[0][0].shape[0]
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

        mTQ = [dot(markers.T, Q) for Q in self._QS[0]]
        yTQDiQTm = [dot(i, j.T) for (i, j) in zip(self._yTQDi, mTQ)]

        XTQDiQTm = [dot(i, j.T) for (i, j) in zip(self._XTQDi, mTQ)]
        mTQDiQTm = [npsum((i / j) * i, axis=1) for (i, j) in zip(mTQ, self._D)]

        nsamples = markers.shape[0]
        nmarkers = markers.shape[1]

        lmls = full(nmarkers, self._static_lml())
        effect_sizes = empty(nmarkers)

        if self._XTQDi[0].shape[0] == 1:
            return self._fast_scan_chunk_1covariate_loop(
                lmls, effect_sizes, yTQDiQTm, XTQDiQTm, mTQDiQTm, nsamples)
        else:
            return self._fast_scan_chunk_multicovariate_loop(
                lmls, effect_sizes, yTQDiQTm, XTQDiQTm, mTQDiQTm, nsamples,
                nmarkers)

    def _fast_scan_chunk_multicovariate_loop(self, lmls, effect_sizes,
                                             yTQDiQTm, XTQDiQTm, mTQDiQTm,
                                             nsamples, nmarkers):
        b00m = empty(len(self._yTQDiQTX[0]) + 1)
        b00m[:-1] = self._yTQDiQTX[0]

        b11m = empty(len(self._yTQDiQTX[1]) + 1)
        b11m[:-1] = self._yTQDiQTX[1]

        for i in range(nmarkers):

            b00m[-1] = yTQDiQTm[0][i]
            b11m[-1] = yTQDiQTm[1][i]

            self._C[0][:-1, -1] = XTQDiQTm[0][:, i]
            self._C[1][:-1, -1] = XTQDiQTm[1][:, i]

            self._C[0][-1, :-1] = self._C[0][:-1, -1]
            self._C[1][-1, :-1] = self._C[1][:-1, -1]

            self._C[0][-1, -1] = mTQDiQTm[0][i]
            self._C[1][-1, -1] = mTQDiQTm[1][i]

            try:
                beta = solve(self._C[1] - self._C[0], b11m - b00m)
            except LinAlgError:
                try:
                    beta = rsolve(self._C[1] - self._C[0], b11m - b00m)
                except LinAlgError:
                    msg = "Could not converge to the optimal"
                    msg += " effect-size of one of the candidates."
                    msg += " Setting its effect-size to zero."
                    logging.getLogger(__name__).warning(msg)
                    beta = zeros(self._C[1].shape[0])

            effect_sizes[i] = beta[-1]

            if self._scale is None:
                # _compute_scale(nsamples, beta, self._yTQDiQTy, self._yTQDiQTX, )
                # (nsamples, beta, yTQDiQTy, yTQDiQTX, yTQDiQTm, XTQDiQTX,
                #                    XTQDiQTm, mTQDiQTm)

                p0 = self._yTQDiQTy[0] - 2 * b00m.dot(beta) + beta.dot(
                    self._C[0]).dot(beta)
                p1 = self._yTQDiQTy[1] - 2 * b11m.dot(beta) + beta.dot(
                    self._C[1].dot(beta))

                scale = (p0 + p1) / nsamples
            else:
                scale = self._scale

            lmls[i] -= nsamples * log(max(scale, epsilon.super_tiny))

        lmls /= 2
        return lmls, effect_sizes

    def _fast_scan_chunk_1covariate_loop(self, lmls, effect_sizes, yTQDiQTm,
                                         XTQDiQTm, mTQDiQTm, nsamples):

        C00 = [C[0, 0] for C in self._C]
        C01 = [c[0, :] for c in XTQDiQTm]

        b = [bi[0] for bi in self._yTQDiQTX]

        sC00 = C00[1] - C00[0]
        sC01 = C01[1] - C01[0]
        sC11 = mTQDiQTm[1] - mTQDiQTm[0]

        sb = b[1] - b[0]
        sbm = yTQDiQTm[1] - yTQDiQTm[0]

        with errstate(divide='ignore'):
            beta = beta_1covariate(sb, sbm, sC00, sC01, sC11)

        beta = [nan_to_num(bet) for bet in beta]

        scale = zeros(len(lmls))

        if self._scale is None:
            for i in range(2):
                scale += self._yTQDiQTy[i] - 2 * (
                    b[i] * beta[0] + yTQDiQTm[i] * beta[1])
                scale += beta[0] * (C00[i] * beta[0] + C01[i] * beta[1])
                scale += beta[1] * (C01[i] * beta[0] + mTQDiQTm[i] * beta[1])
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


def beta_1covariate(sb, sbm, sC00, sC01, sC11):
    d0 = sb / sC00
    d1 = sb / sC01

    d3 = sbm / sC01
    d4 = sbm / sC11

    d5 = sC00 / sC01
    d6 = sC11 / sC01

    return [(d1 - d4) / (d5 - 1 / d6), (-d0 + d3) / (d6 - 1 / d5)]


def _compute_scale(nsamples, beta, yTQDiQTy, yTQDiQTX, yTQDiQTm, XTQDiQTX,
                   XTQDiQTm, mTQDiQTm):

    scale = npsum(yTQDiQTy[i] for i in range(2))
    scale -= npsum(2 * yTQDiQTX[i] * beta[0] for i in range(2))
    scale -= npsum(2 * yTQDiQTm[i] * beta[1] for i in range(2))

    scale += npsum(beta[0] * XTQDiQTX[i] * beta[0] for i in range(2))
    scale += npsum(beta[0] * XTQDiQTm[i] * beta[1] for i in range(2))
    scale += npsum(beta[1] * XTQDiQTm[i] * beta[0] for i in range(2))
    scale += npsum(beta[1] * mTQDiQTm[i] * beta[1] for i in range(2))

    return scale / nsamples
