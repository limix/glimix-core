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

LOG2PI = 1.837877066409345339081937709124758839607238769531250


class FastScanner(object):
    r"""Approximated fast inference over several covariates.

    Let :math:`\mathrm X` a samples-by-covariates matrix,
    :math:`\mathrm M` a samples-by-markers matrix, and
    :math:`\mathbf y` an array of outcome.
    A covariance :math:`\mathrm K` will be provided via its economic eigen
    decomposition ``((Q0, Q1), S0)`` and ``delta`` will define the ratio of
    covariation between :math:`\mathrm K` and the identity in the formulae

    .. math::

        \mathbf y \sim \mathcal N\big(~ \mathrm X\boldsymbol\beta
        + \mathrm{M}_i \alpha_i;~
        s ((1 - \delta) \mathrm K + \delta\mathrm I) ~\big)


    Note that :math:`\alpha_i` is a scalar multiplying a column-matrix
    :math:`\mathrm{M}_i`.
    The variable :math:`s` is a scalar that is jointly adjusted with
    :math:`\alpha_i` in order the maximize the marginal likelihood, ultimately
    providing the degree of association between the marker :math:`\mathrm{M}_i`
    with the outcome :math:`\mathbf y` via an p-value.
    As mentioned before, the ratio :math:`\delta` is not adjusted for
    performance reason.

    Parameters
    ----------
    y : array_like
        Real-valued outcome.
    X : array_like
        Matrix of covariates.
    QS : tuple
        Economic eigen decomposition ``((Q0, Q1), S0)``.
    iid_variance : float
        Variance due to iid effect..
    """

    def __init__(self, y, X, QS, iid_variance):

        self._QS = QS
        self._diags = [QS[1] + iid_variance, iid_variance]

        yTQ = [dot(y.T, Q) for Q in QS[0]]

        XTQ = [dot(X.T, Q) for Q in QS[0]]

        self._yTQdiag = [l / r for (l, r) in zip(yTQ, self._diags)]

        self._a = [(i**2 / j).sum() for (i, j) in zip(yTQ, self._diags)]

        self._b = [dot(i, j.T) for (i, j) in zip(self._yTQdiag, XTQ)]

        self._XTQdiag = [i / j for (i, j) in zip(XTQ, self._diags)]

        nc = X.shape[1]

        self._C = [empty((nc + 1, nc + 1)), empty((nc + 1, nc + 1))]

        for i in range(2):
            self._C[i][:-1, :-1] = dot(self._XTQdiag[i], XTQ[i].T)

    def null_lml(self):
        # TODO
        # def _diag(self, i):
        #     if i == 0:
        #         return self._QS[1] * (1 - self.delta) + self.delta
        #     return self.delta

        n = len(self._y)
        p = n - self._QS[1].shape[0]
        LOG2PI = 1.837877066409345339081937709124758839607238769531250
        lml = -n * LOG2PI - n - n * log(self.scale)
        lml += -sum(log(self._diags[0])) - p * log(self._diags[1])
        lml /= 2
        return lml

    def fast_scan(self, markers, verbose=True):
        r"""LMLs of markers by fitting scale and fixed-effect sizes parameters.

        Parameters
        ----------
        markers : array_like
            Matrix of fixed-effects across columns.
        verbose : bool, optional
            ``True`` for progress information; ``False`` otherwise.
            Defaults to ``True``.

        Returns
        -------
        array_like : LMLs.
        array_like : Effect-sizes.
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

    def _static_lml(self):
        n = self._QS[0][0].shape[0]
        p = len(self._diags[0])
        static_lml = -n * LOG2PI - n
        static_lml -= npsum(log(self._diags[0]))
        static_lml -= (n - p) * log(self._diags[1])
        return static_lml

    def _fast_scan_chunk(self, markers):
        markers = asarray(markers, float)

        if not markers.ndim == 2:
            raise ValueError("`markers` array must be bidimensional.")

        if not npall(isfinite(markers)):
            raise ValueError("One or more variants have non-finite value.")

        mTQ = [dot(markers.T, Q) for Q in self._QS[0]]
        bm = [dot(i, j.T) for (i, j) in zip(self._yTQdiag, mTQ)]

        c_01 = [dot(i, j.T) for (i, j) in zip(self._XTQdiag, mTQ)]
        c_11 = [npsum((i / j) * i, axis=1) for (i, j) in zip(mTQ, self._diags)]

        nsamples = markers.shape[0]
        nmarkers = markers.shape[1]

        lmls = full(nmarkers, self._static_lml())
        effect_sizes = empty(nmarkers)

        if self._XTQdiag[0].shape[0] == 1:
            return self._fast_scan_chunk_1covariate_loop(
                lmls, effect_sizes, bm, c_01, c_11, nsamples)
        else:
            return self._fast_scan_chunk_multicovariate_loop(
                lmls, effect_sizes, bm, c_01, c_11, nsamples, nmarkers)

    def _fast_scan_chunk_multicovariate_loop(self, lmls, effect_sizes, bm,
                                             c_01, c_11, nsamples, nmarkers):
        b00m = empty(len(self._b[0]) + 1)
        b00m[:-1] = self._b[0]

        b11m = empty(len(self._b[1]) + 1)
        b11m[:-1] = self._b[1]

        for i in range(nmarkers):

            b00m[-1] = bm[0][i]
            b11m[-1] = bm[1][i]

            self._C[0][:-1, -1] = c_01[0][:, i]
            self._C[1][:-1, -1] = c_01[1][:, i]

            self._C[0][-1, :-1] = self._C[0][:-1, -1]
            self._C[1][-1, :-1] = self._C[1][:-1, -1]

            self._C[0][-1, -1] = c_11[0][i]
            self._C[1][-1, -1] = c_11[1][i]

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

            p0 = self._a[0] - 2 * b00m.dot(beta) + beta.dot(
                self._C[0]).dot(beta)
            p1 = self._a[1] - 2 * b11m.dot(beta) + beta.dot(
                self._C[1].dot(beta))

            scale = (p0 + p1) / nsamples

            lmls[i] -= nsamples * log(max(scale, epsilon.super_tiny))

        lmls /= 2
        return lmls, effect_sizes

    def _fast_scan_chunk_1covariate_loop(self, lmls, effect_sizes, bm, c_01,
                                         C11, nsamples):

        C00 = [C[0, 0] for C in self._C]
        C01 = [c[0, :] for c in c_01]

        b = [bi[0] for bi in self._b]

        sC00 = C00[1] - C00[0]
        sC01 = C01[1] - C01[0]
        sC11 = C11[1] - C11[0]

        sb = b[1] - b[0]
        sbm = bm[1] - bm[0]

        with errstate(divide='ignore'):
            beta = beta_1covariate(sb, sbm, sC00, sC01, sC11)

        beta = [nan_to_num(bet) for bet in beta]

        scale = zeros(len(lmls))

        for i in range(2):
            scale += self._a[i] - 2 * (b[i] * beta[0] + bm[i] * beta[1])
            scale += beta[0] * (C00[i] * beta[0] + C01[i] * beta[1])
            scale += beta[1] * (C01[i] * beta[0] + C11[i] * beta[1])

        scale /= nsamples

        lmls -= nsamples * log(clip(scale, epsilon.super_tiny, inf))
        lmls /= 2

        effect_sizes = beta[1]

        return lmls, effect_sizes


def beta_1covariate(sb, sbm, sC00, sC01, sC11):
    d0 = sb / sC00
    d1 = sb / sC01

    d3 = sbm / sC01
    d4 = sbm / sC11

    d5 = sC00 / sC01
    d6 = sC11 / sC01

    return [(d1 - d4) / (d5 - 1 / d6), (-d0 + d3) / (d6 - 1 / d5)]
