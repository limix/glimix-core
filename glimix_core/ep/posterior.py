from __future__ import division

from numpy import sum as npsum
from numpy import dot, empty, sqrt
from numpy_sugar.linalg import cho_solve, ddot, dotd, sum2diag
from scipy.linalg import cho_factor


class Posterior(object):
    r"""EP posterior.

    It is given by

    .. math::

        \mathbf z \sim \mathcal N\left(\Sigma (\tilde{\mathrm T}
          \tilde{\boldsymbol\mu} + \mathrm K^{-1}\mathbf m),
          (\tilde{\mathrm T} + \mathrm K^{-1})^{-1}\right).
    """

    def __init__(self, site):
        n = len(site.tau)
        self.tau = empty(n)
        self.eta = empty(n)
        self._site = site
        self._mean = None
        self._cov = None

    def _initialize(self):
        r"""Initialize the mean and covariance of the posterior.

        Given that :math:`\tilde{\mathrm T}` is a matrix of zeros right before
        the first EP iteration, we have

        .. math::

            \boldsymbol\mu = \mathrm K^{-1} \mathbf m ~\text{ and }~
            \Sigma = \mathrm K

        as the initial posterior mean and covariance.
        """
        if self._mean is None or self._cov is None:
            return

        Q = self._cov['QS'][0][0]
        S = self._cov['QS'][1]

        self.tau[:] = 1 / npsum((Q * sqrt(S))**2, axis=1)
        self.eta[:] = self._mean
        self.eta[:] *= self.tau

    @property
    def mean(self):
        self._initialize()
        return self._mean

    @mean.setter
    def mean(self, v):
        self._mean = v

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, v):
        self._initialize()
        self._cov = v

    def L(self):
        r"""Cholesky decomposition of :math:`\mathrm B`.

        .. math::

            \mathrm B = \mathrm Q^{\intercal}\tilde{\mathrm{T}}\mathrm Q
                + \mathrm{S}^{-1}
        """
        Q = self._cov['QS'][0][0]
        S = self._cov['QS'][1]
        B = dot(Q.T, ddot(self._site.tau, Q, left=True))
        sum2diag(B, 1. / S, out=B)
        return cho_factor(B, lower=True)[0]

    def _BiQt(self):
        Q = self._cov['QS'][0][0]
        return cho_solve(self.L(), Q.T)

    def update(self):
        Q = self._cov['QS'][0][0]
        S = self._cov['QS'][1]

        K = dot(ddot(Q, S, left=False), Q.T)

        BiQt = self._BiQt()
        TK = ddot(self._site.tau, K, left=True)
        BiQtTK = dot(BiQt, TK)

        self.tau[:] = K.diagonal()
        self.tau -= dotd(Q, BiQtTK)
        self.tau[:] = 1 / self.tau

        if not all(self.tau >= 0.):
            raise RuntimeError("'tau' has to be non-negative.")

        self.eta[:] = dot(K, self._site.eta)
        self.eta[:] += self._mean
        self.eta[:] -= dot(Q, dot(BiQtTK, self._site.eta))
        self.eta[:] -= dot(Q, dot(BiQt, self._site.tau * self._mean))

        self.eta *= self.tau
