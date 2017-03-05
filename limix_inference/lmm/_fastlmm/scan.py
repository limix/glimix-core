from __future__ import division

from numpy import asarray, concatenate, dot, log, newaxis

from numpy_sugar.linalg import solve


class NormalLikTrick(object):
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

        lmls = []
        for marker in markers.T:

            MTQ0 = dot(
                concatenate([self.M, marker[:, newaxis]], axis=1).T, self.Q0)
            MTQ1 = dot(
                concatenate([self.M, marker[:, newaxis]], axis=1).T, self.Q1)

            b0 = (self.yTQ0 / self.diag0).dot(MTQ0.T)
            c0 = (MTQ0 / self.diag0).dot(MTQ0.T)

            b1 = (self.yTQ1 / self.diag1).dot(MTQ1.T)
            c1 = (MTQ1 / self.diag1).dot(MTQ1.T)

            nominator = b1 - b0
            denominator = c1 - c0

            beta = solve(denominator, nominator)

            p0 = self.a1 - 2 * b1.dot(beta) + beta.dot(c1.dot(beta))
            p1 = self.a0 - 2 * b0.dot(beta) + beta.dot(c0).dot(beta)

            n = markers.shape[0]

            scale = (p0 + p1) / n

            LOG2PI = 1.837877066409345339081937709124758839607238769531250
            lml = -n * LOG2PI - n - n * log(scale)
            lml += -sum(log(self.diag0)) - (n - len(self.diag0)
                                            ) * log(self.diag1)
            lml /= 2
            lmls.append(lml)

        return asarray(lmls, float)
