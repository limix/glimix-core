from __future__ import division

from numpy import append, dot, empty, log
from numpy import sum as npsum

from numpy_sugar.linalg import solve, ddot


class FastLMMScanner(object):
    r"""Necessary data for fast lml computation over many fixed-effects."""

    def __init__(self, M, Q0, Q1, yTQ0, yTQ1, diag0, diag1, a0, a1):
        self.M = M
        self.Q0 = Q0
        self.Q1 = Q1
        self.yTQ0 = yTQ0
        self.yTQ1 = yTQ1
        self.diag0 = diag0
        self.diag1 = diag1
        self.a0 = a0
        self.a1 = a1
        self.transform = None

    def fast_scan(self, markers):
        r"""LMLs of markers by fitting scale and fixed-effect sizes parameters.

        The likelihood is given by

        .. math::

            \mathcal N\big(~\mathbf y ~|~ \boldsymbol\beta^{\intercal}
            [\mathrm M ~~ \tilde{\mathrm M}],  s \mathrm K~\big),

        where :math:`s` is the scale parameter and :math:`\boldsymbol\beta` is the
        fixed-effect sizes; :math:`\tilde{\mathrm M}` is a marker to be scanned.
        """

        if self.transform is not None:
            if self.transform.ndim == 1:
                markers = ddot(self.transform, markers, left=True)
            else:
                markers = dot(self.transform, markers)

        assert markers.ndim == 2

        nc = self.M.shape[1]
        M = self.M

        MTQ0 = dot(M.T, self.Q0)
        mTQ0 = dot(markers.T, self.Q0)

        MTQ1 = dot(M.T, self.Q1)
        mTQ1 = dot(markers.T, self.Q1)

        yTQ0diag0 = self.yTQ0 / self.diag0
        yTQ1diag1 = self.yTQ1 / self.diag1

        b0 = yTQ0diag0.dot(MTQ0.T)
        b0m = yTQ0diag0.dot(mTQ0.T)

        MTQ0diag0 = MTQ0 / self.diag0
        MTQ1diag1 = MTQ1 / self.diag1

        c0_00 = MTQ0diag0.dot(MTQ0.T)
        c0_01 = MTQ0diag0.dot(mTQ0.T)
        c0_11 = npsum((mTQ0 / self.diag0) * mTQ0, axis=1)

        b1 = yTQ1diag1.dot(MTQ1.T)
        b1m = yTQ1diag1.dot(mTQ1.T)

        c1_00 = MTQ1diag1.dot(MTQ1.T)
        c1_01 = MTQ1diag1.dot(mTQ1.T)
        c1_11 = npsum((mTQ1 / self.diag1) * mTQ1, axis=1)

        C0 = empty((nc + 1, nc + 1))
        C0[:-1, :-1] = c0_00

        C1 = empty((nc + 1, nc + 1))
        C1[:-1, :-1] = c1_00

        n = markers.shape[0]
        lmls = empty(markers.shape[1])
        effect_sizes = empty(markers.shape[1])

        LOG2PI = 1.837877066409345339081937709124758839607238769531250
        lmls[:] = -n * LOG2PI - n
        lmls[:] += -npsum(log(self.diag0)) - (n - len(self.diag0)
                                              ) * log(self.diag1)

        for i in range(markers.shape[1]):

            b11m = append(b1, b1m[i])
            b00m = append(b0, b0m[i])

            nominator = b11m - b00m

            C0[:-1, -1] = c0_01[:, i]
            C1[:-1, -1] = c1_01[:, i]

            C0[-1, :-1] = C0[:-1, -1]
            C1[-1, :-1] = C1[:-1, -1]

            C0[-1, -1] = c0_11[i]
            C1[-1, -1] = c1_11[i]

            denominator = C1 - C0

            beta = solve(denominator, nominator)
            effect_sizes[i] = beta[-1]

            p0 = self.a1 - 2 * b11m.dot(beta) + beta.dot(C1.dot(beta))
            p1 = self.a0 - 2 * b00m.dot(beta) + beta.dot(C0).dot(beta)

            scale = (p0 + p1) / n

            lmls[i] -= n * log(scale)

        lmls /= 2
        return lmls, effect_sizes
