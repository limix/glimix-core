from __future__ import division

from numpy import dot
from scipy.linalg import cho_factor

from numpy_sugar.linalg import cho_solve, ddot, dotd, sum2diag

from .posterior import Posterior


def _cho_factor(B):
    B = cho_factor(B, overwrite_a=True, lower=True, check_finite=False)[0]
    return B


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
        if self._L_cache is not None:
            return self._L_cache

        s = self._cov['scale']
        d = self._cov['delta']
        Q = self._cov['QS'][0][0]
        S = self._cov['QS'][1]

        ddot(self._A * self._site.tau, Q, left=True, out=self._NxR)
        B = dot(Q.T, self._NxR, out=self._RxR)
        B *= 1 - d
        sum2diag(B, 1. / S / s, out=B)
        self._L_cache = _cho_factor(B)
        return self._L_cache

    def update(self):
        self._L_cache = None
        self._LQT_cache = None

        s = self._cov['scale']
        d = self._cov['delta']
        Q = self._cov['QS'][0][0]
        S = self._cov['QS'][1]

        A = self._A
        L = self.L()

        t = self._site.tau
        e = self._site.eta

        QA = ddot(Q.T, A, out=self._RxN)
        QS0 = ddot(Q, S, out=self._NxR)

        tBitQt = cho_solve(L, Q.T)
        tBitQtA = ddot(tBitQt, A)
        ATtQ = ddot(A * t, Q)
        QtATtQ = dot(ddot(QA, t), Q)

        self.tau[:] = s * (1 - d) * dotd(QS0, QA)
        D = dot(QtATtQ, tBitQtA)
        self.tau -= s * (1 - d) * dotd(QS0, D) * (1 - d)
        self.tau += s * d * A
        self.tau -= s * d * dotd(ATtQ, tBitQtA) * (1 - d)
        self.tau **= -1

        v = s * (1 - d) * dot(Q, S * dot(Q.T, e)) + s * d * e + self._mean

        self.eta[:] = A * v
        self.eta -= A * dot(Q, dot(tBitQtA, t * v)) * (1 - d)
        self.eta *= self.tau
