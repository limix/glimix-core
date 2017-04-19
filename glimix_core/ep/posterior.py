from __future__ import division

from numpy import dot, empty, sqrt
from numpy import sum as npsum
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
        self._QS = None

    def set_prior_mean(self, mean):
        self._mean = mean

    def set_prior_cov(self, QS):
        self._QS = QS

    def prior_mean(self):
        return self._mean

    def prior_cov(self):
        return self._QS

    def initialize(self):
        r"""Initialize the mean and covariance of the posterior.

        Given that :math:`\tilde{\mathrm T}` is a matrix of zeros right before
        the first EP iteration, we have

        .. math::

            \boldsymbol\mu = \mathrm K^{-1} \mathbf m ~\text{ and }~
            \Sigma = \mathrm K

        as the initial posterior mean and covariance.
        """
        self.tau[:] = 1 / npsum((self._QS[0] * sqrt(self._QS[1]))**2,
                                axis=1)
        self.eta[:] = self._mean
        self.eta[:] *= self.tau

    def L(self):
        r"""Cholesky decomposition of :math:`\mathrm B`.

        .. math::

            \mathrm B = \mathrm Q^{\intercal}\tilde{\mathrm{T}}\mathrm Q
                + \mathrm{S}^{-1}
        """
        QS = self._QS
        B = dot(QS[0].T, ddot(self._site.tau, QS[0], left=True))
        sum2diag(B, 1. / QS[1], out=B)
        return cho_factor(B, lower=True)[0]

    # def QS(self):
    #     r"""Eigen decomposition of :math:`\mathrm K`"""
    #     QS = economic_qs(self._cov)
    #     return QS[0][0], QS[1]

    def _BiQt(self):
        return cho_solve(self.L(), self._QS[0].T)

    def update(self):
        QS = self._QS
        cov = dot(ddot(QS[0], QS[1], left=False), QS[0].T)

        BiQt = self._BiQt()
        TK = ddot(self._site.tau, cov, left=True)
        BiQtTK = dot(BiQt, TK)

        self.tau[:] = cov.diagonal()
        self.tau -= dotd(QS[0], BiQtTK)
        self.tau[:] = 1/self.tau

        assert all(self.tau >= 0.)

        self.eta[:] = dot(cov, self._site.eta)
        self.eta[:] += self._mean
        self.eta[:] -= dot(QS[0], dot(BiQtTK, self._site.eta))
        self.eta[:] -= dot(QS[0], dot(BiQt, self._site.tau * self._mean))

        self.eta *= self.tau
