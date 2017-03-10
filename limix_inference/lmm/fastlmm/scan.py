from __future__ import division

from numpy import append, dot, empty, log, full
from numpy import sum as npsum

from numpy_sugar.linalg import solve


class FastLMMScanner(object):
    r"""Fast inference over multiple covariates.

    .. math::

        aw
    """

    def __init__(self, y, M, QS, delta):

        self._QS = QS

        self._diag0 = (1 - delta) * QS[1] + delta
        self._delta = delta

        yTQ0 = dot(y.T, QS[0][0])

        yTQ1 = dot(y.T, QS[0][1])

        MTQ0 = dot(M.T, QS[0][0])
        MTQ1 = dot(M.T, QS[0][1])

        self._yTQ0diag0 = yTQ0 / self._diag0
        self._yTQ1diag1 = yTQ1 / delta

        self._a0 = (yTQ0**2 / self._diag0).sum()
        self._b0 = dot(self._yTQ0diag0, MTQ0.T)

        self._a1 = (yTQ1**2).sum() / delta
        self._b1 = dot(self._yTQ1diag1, MTQ1.T)

        self._MTQ0diag0 = MTQ0 / self._diag0
        self._MTQ1diag1 = MTQ1 / delta

        LOG2PI = 1.837877066409345339081937709124758839607238769531250
        n = len(y)
        self._static_lml = -n * LOG2PI - n
        self._static_lml -= npsum(log(self._diag0))
        self._static_lml -= (n - len(self._diag0)) * log(delta)

        nc = M.shape[1]

        self._C0 = empty((nc + 1, nc + 1))
        self._C0[:-1, :-1] = dot(self._MTQ0diag0, MTQ0.T)

        self._C1 = empty((nc + 1, nc + 1))
        self._C1[:-1, :-1] = dot(self._MTQ1diag1, MTQ1.T)

    def fast_scan(self, markers):
        r"""LMLs of markers by fitting scale and fixed-effect sizes parameters.

        The likelihood is given by

        .. math::

            \mathcal N\big(~\mathbf y ~|~ \boldsymbol\beta^{\intercal}
            [\mathrm M ~~ \tilde{\mathrm M}],  s \mathrm K~\big),

        where :math:`s` is the scale parameter and :math:`\boldsymbol\beta` is the
        fixed-effect sizes; :math:`\tilde{\mathrm M}` is a marker to be scanned.
        """
        assert markers.ndim == 2

        mTQ0 = dot(markers.T, self._QS[0][0])
        mTQ1 = dot(markers.T, self._QS[0][1])

        b0m = dot(self._yTQ0diag0, mTQ0.T)
        b1m = dot(self._yTQ1diag1, mTQ1.T)

        c0_01 = dot(self._MTQ0diag0, mTQ0.T)
        c0_11 = npsum((mTQ0 / self._diag0) * mTQ0, axis=1)

        c1_01 = dot(self._MTQ1diag1, mTQ1.T)
        c1_11 = npsum((mTQ1 / self._delta) * mTQ1, axis=1)

        n = markers.shape[0]
        lmls = full(markers.shape[1], self._static_lml)
        effect_sizes = empty(markers.shape[1])

        for i in range(markers.shape[1]):

            b00m = append(self._b0, b0m[i])
            b11m = append(self._b1, b1m[i])

            nominator = b11m - b00m

            self._C0[:-1, -1] = c0_01[:, i]
            self._C1[:-1, -1] = c1_01[:, i]

            self._C0[-1, :-1] = self._C0[:-1, -1]
            self._C1[-1, :-1] = self._C1[:-1, -1]

            self._C0[-1, -1] = c0_11[i]
            self._C1[-1, -1] = c1_11[i]

            denominator = self._C1 - self._C0

            beta = solve(denominator, nominator)
            effect_sizes[i] = beta[-1]

            p0 = self._a1 - 2 * b11m.dot(beta) + beta.dot(self._C1.dot(beta))
            p1 = self._a0 - 2 * b00m.dot(beta) + beta.dot(self._C0).dot(beta)

            scale = (p0 + p1) / n

            lmls[i] -= n * log(scale)

        lmls /= 2
        return lmls, effect_sizes
