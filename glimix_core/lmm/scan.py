from __future__ import division

import logging

from numpy import sum as npsum
from numpy import append, dot, empty, full, log, zeros
from numpy.linalg import LinAlgError
from numpy_sugar.linalg import rsolve, solve
from tqdm import tqdm

LOG2PI = 1.837877066409345339081937709124758839607238769531250


class FastScanner(object):  # pylint: disable=R0903
    r"""Fast inference over multiple covariates.

    Let :math:`\tilde{\mathrm M}_i` be a column-matrix of fixed-effect
    :math:`i`.
    It fits :math:`\alpha_i` and refits :math:`\boldsymbol\beta` and :math:`s`
    for each fixed-effect :math:`i` in order to compute LMLs:

    .. math::

        \mathbf y \sim \mathcal N\big(~ \mathrm M\boldsymbol\beta
        + \tilde{\mathrm M}_i \alpha_i;~ s \mathrm K ~\big)

    Args:

        y (array_like): real-valued outcome.
        M (array_like): matrix of covariates.
        QS (tuple): economic eigen decomposition ``((Q0, Q1), S0)``.
    """

    def __init__(self, y, M, QS, delta):

        self._QS = QS
        self._diags = [(1 - delta) * QS[1] + delta, delta]
        self._logger = logging.getLogger(__name__)

        yTQ = [dot(y.T, Q) for Q in QS[0]]
        MTQ = [dot(M.T, Q) for Q in QS[0]]

        self._yTQdiag = [l / r for (l, r) in zip(yTQ, self._diags)]

        self._a = [(i**2 / j).sum() for (i, j) in zip(yTQ, self._diags)]
        self._b = [dot(i, j.T) for (i, j) in zip(self._yTQdiag, MTQ)]

        self._MTQdiag = [i / j for (i, j) in zip(MTQ, self._diags)]

        nc = M.shape[1]

        self._C = [empty((nc + 1, nc + 1)), empty((nc + 1, nc + 1))]

        for i in range(2):
            self._C[i][:-1, :-1] = dot(self._MTQdiag[i], MTQ[i].T)

    def _static_lml(self):
        n = self._QS[0][0].shape[0]
        p = len(self._diags[0])
        static_lml = -n * LOG2PI - n
        static_lml -= npsum(log(self._diags[0]))
        static_lml -= (n - p) * log(self._diags[1])
        return static_lml

    def fast_scan(self, markers, verbose=True):
        r"""LMLs of markers by fitting scale and fixed-effect sizes parameters.

        Args:

            markers (array_like): matrix of fixed-effects across columns.

        Returns:
            tuple: LMLs and effect-sizes, respectively.
        """
        assert markers.ndim == 2
        p = markers.shape[1]

        lmls = empty(p)
        effect_sizes = empty(p)

        nchunks = min(p, 30)
        chunk_size = (p + nchunks - 1) // nchunks

        for i in tqdm(range(nchunks), desc="Scanning", disable=not verbose):
            start = i * chunk_size
            stop = min(start + chunk_size, markers.shape[1])

            l, e = self._fast_scan_chunk(markers[:, start:stop])

            lmls[start:stop] = l
            effect_sizes[start:stop] = e

        return lmls, effect_sizes

    def _fast_scan_chunk(self, markers):
        assert markers.ndim == 2

        mTQ = [dot(markers.T, Q) for Q in self._QS[0]]

        bm = [dot(i, j.T) for (i, j) in zip(self._yTQdiag, mTQ)]

        c_01 = [dot(i, j.T) for (i, j) in zip(self._MTQdiag, mTQ)]
        c_11 = [npsum((i / j) * i, axis=1) for (i, j) in zip(mTQ, self._diags)]

        lmls = full(markers.shape[1], self._static_lml())
        effect_sizes = empty(markers.shape[1])

        for i in range(markers.shape[1]):

            b00m = append(self._b[0], bm[0][i])
            b11m = append(self._b[1], bm[1][i])

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
                    self._logger.warning(warn)
                    beta = zeros(self._C[1].shape[0])

            effect_sizes[i] = beta[-1]

            p0 = self._a[1] - 2 * b11m.dot(beta) + beta.dot(
                self._C[1].dot(beta))
            p1 = self._a[0] - 2 * b00m.dot(beta) + beta.dot(
                self._C[0]).dot(beta)

            scale = (p0 + p1) / markers.shape[0]

            lmls[i] -= markers.shape[0] * log(scale)

        lmls /= 2
        return lmls, effect_sizes
