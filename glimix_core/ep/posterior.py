from __future__ import division

from numpy import sum as npsum
from numpy import dot, empty, sqrt, concatenate, zeros
from scipy.linalg import cho_factor

from numpy_sugar.linalg import cho_solve, ddot, dotd, sum2diag


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

    def set_prior_cov(self, cov):
        self._QS = cov['QS']

    def prior_mean(self):
        return self._mean

    def prior_cov(self):
        return (self._QS[0][0], self._QS[1])

    def initialize(self):
        r"""Initialize the mean and covariance of the posterior.

        Given that :math:`\tilde{\mathrm T}` is a matrix of zeros right before
        the first EP iteration, we have

        .. math::

            \boldsymbol\mu = \mathrm K^{-1} \mathbf m ~\text{ and }~
            \Sigma = \mathrm K

        as the initial posterior mean and covariance.
        """
        #self.tau[:] = 1 / npsum((self._QS[0] * sqrt(self._QS[1]))**2, axis=1)
        self.tau[:] = 1 / npsum((self._QS[0][0] * sqrt(self._QS[1]))**2, axis=1)
        self.eta[:] = self._mean
        self.eta[:] *= self.tau

    def L(self):
        r"""Cholesky decomposition of :math:`\mathrm B`.

        .. math::

            \mathrm B = \mathrm Q^{\intercal}\tilde{\mathrm{T}}\mathrm Q
                + \mathrm{S}^{-1}
        """
        QS = self._QS
        #B = dot(QS[0].T, ddot(self._site.tau, QS[0], left=True))
        B = dot(QS[0][0].T, ddot(self._site.tau, QS[0][0], left=True))
        sum2diag(B, 1. / QS[1], out=B)
        return cho_factor(B, lower=True)[0]

    def _BiQt(self):
        return cho_solve(self.L(), self._QS[0][0].T)

    def update(self):
        QS = self._QS
        cov = dot(ddot(QS[0][0], QS[1], left=False), QS[0][0].T)

        BiQt = self._BiQt()
        TK = ddot(self._site.tau, cov, left=True)
        BiQtTK = dot(BiQt, TK)

        self.tau[:] = cov.diagonal()
        self.tau -= dotd(QS[0][0], BiQtTK)
        self.tau[:] = 1 / self.tau

        assert all(self.tau >= 0.)

        self.eta[:] = dot(cov, self._site.eta)
        self.eta[:] += self._mean
        self.eta[:] -= dot(QS[0][0], dot(BiQtTK, self._site.eta))
        self.eta[:] -= dot(QS[0][0], dot(BiQt, self._site.tau * self._mean))

        self.eta *= self.tau


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
        self._scale = None
        self._delta = None
        self._QS0Qt = None

    def set_prior_cov(self, cov):
        if self._QS is None:
            self._QS = cov['QS']
            self._QS0 = ddot(self._QS[0][0],
                             self._QS[1], left=False)

        self._scale = cov['scale']
        self._delta = cov['delta']
        # self._S = self._scale * ((1 - self._delta) * self._QS[1] + self._delta)

    @property
    def _A(self):
        return 1 / (self._scale * self._delta * self._site.tau + 1)

    def prior_cov(self):
        return dict(QS=self._QS, scale=self._scale, delta=self._delta)

    def initialize(self):
        r"""Initialize the mean and covariance of the posterior.

        Given that :math:`\tilde{\mathrm T}` is a matrix of zeros right before
        the first EP iteration, we have

        .. math::

            \boldsymbol\mu = \mathrm K^{-1} \mathbf m ~\text{ and }~
            \Sigma = \mathrm K

        as the initial posterior mean and covariance.
        """
        s = self._scale
        d = self._delta
        Q = self._QS[0][0]
        S = self._QS[1]
        self.tau[:] = s * (1 - d) * npsum((Q * sqrt(S))**2, axis=1)
        self.tau += s * d
        self.tau **= -1
        self.eta[:] = self._mean
        self.eta[:] *= self.tau

    def L(self):
        r"""Cholesky decomposition of :math:`\mathrm B`.

        .. math::

            \mathrm B = \mathrm Q^{\intercal}\tilde{\mathrm{T}}\mathrm Q
                + \mathrm{S}^{-1}
        """
        A = self._A
        tQ = sqrt(1 - self._delta) * self._QS[0][0]
        S = self._QS[1]
        B = dot(tQ.T, ddot(A * self._site.tau, tQ, left=True))
        sum2diag(B, 1. / S / self._scale, out=B)
        return cho_factor(B, lower=True)[0]

    def update(self):
        Q = self._QS[0][0]
        S = self._QS[1]
        A = self._A

        s = self._scale
        d = self._delta

        tQ = sqrt(1 - d) * Q
        QtA = ddot(Q.T, A, left=False)

        tBitQt = cho_solve(self.L(), tQ.T)

        tBitQtA = ddot(tBitQt, A, left=False)

        ATtQ = ddot(A * self._site.tau, tQ, left=True)
        QtATtQ = dot(ddot(QtA, self._site.tau, left=False), tQ)

        self.tau[:] = s * (1 - d) * dotd(self._QS0, QtA)
        self.tau -= s * (1 - d) * dotd(self._QS0, dot(QtATtQ, tBitQtA))
        self.tau += s * d * A
        self.tau -= s * d * dotd(ATtQ, tBitQtA)
        self.tau **= -1

        t = self._site.tau
        e = self._site.eta

        v = s * (1 - d) * dot(Q, S * dot(Q.T, e)) + s * d * e + self._mean

        self.eta[:] = A * v
        self.eta -= A * dot(tQ, dot(tBitQtA, t * v))
        self.eta *= self.tau
