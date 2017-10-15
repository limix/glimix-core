from __future__ import division

from numpy import sum as npsum
from numpy import dot, sqrt
from numpy_sugar.linalg import cho_solve, ddot, dotd, sum2diag
from scipy.linalg import cho_factor

from .posterior import Posterior


class PosteriorLinearKernel(Posterior):
    r"""EP posterior.

    It is given by

    .. math::

        \mathbf z \sim \mathcal N\left(\Sigma (\tilde{\mathrm T}
          \tilde{\boldsymbol\mu} + \mathrm K^{-1}\mathbf m),
          (\tilde{\mathrm T} + \mathrm K^{-1})^{-1}\right).
    """

    def __init__(self, site):
        super(PosteriorLinearKernel, self).__init__(site)

    @property
    def _A(self):
        s = self._cov['scale']
        d = self._cov['delta']
        return 1 / (s * d * self._site.tau + 1)

    def L(self):
        r"""Cholesky decomposition of :math:`\mathrm B`.

        .. math::

            \mathrm B = \mathrm Q^{\intercal}\tilde{\mathrm{T}}\mathrm Q
                + \mathrm{S}^{-1}
        """
        s = self._cov['scale']
        d = self._cov['delta']
        Q = self._cov['QS'][0][0]
        S = self._cov['QS'][1]

        tQ = sqrt(1 - d) * Q
        B = dot(tQ.T, ddot(self._A * self._site.tau, tQ, left=True))
        sum2diag(B, 1. / S / s, out=B)
        return cho_factor(B, lower=True)[0]

    def update(self):
        s = self._cov['scale']
        d = self._cov['delta']
        Q = self._cov['QS'][0][0]
        S = self._cov['QS'][1]

        A = self._A

        t = self._site.tau
        e = self._site.eta

        tQ = sqrt(1 - d) * Q
        QtA = ddot(Q.T, A, left=False)

        tBitQt = cho_solve(self.L(), tQ.T)

        tBitQtA = ddot(tBitQt, A, left=False)

        ATtQ = ddot(A * t, tQ, left=True)
        QtATtQ = dot(ddot(QtA, t, left=False), tQ)

        QS0 = ddot(Q, S, left=False)

        self.tau[:] = s * (1 - d) * dotd(QS0, QtA)
        self.tau -= s * (1 - d) * dotd(QS0, dot(QtATtQ, tBitQtA))
        self.tau += s * d * A
        self.tau -= s * d * dotd(ATtQ, tBitQtA)
        self.tau **= -1

        v = s * (1 - d) * dot(Q, S * dot(Q.T, e)) + s * d * e + self._mean

        self.eta[:] = A * v
        self.eta -= A * dot(tQ, dot(tBitQtA, t * v))
        self.eta *= self.tau
