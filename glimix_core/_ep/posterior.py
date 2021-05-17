from numpy import dot, empty, empty_like, sqrt, sum as npsum


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
        self._NxR_data = None
        self._RxN_data = None
        self._RxR_data = None
        self._L_cache = None
        self._LQt_cache = None
        self._QS_cache = None
        self._AQ_cache = None
        self._TAQ_cache = None
        self._QSQtATQLQtA_cache = None

    @property
    def _NxR(self):
        if self._NxR_data is None:
            Q = self._cov["QS"][0][0]
            self._NxR_data = empty_like(Q)
        return self._NxR_data

    @property
    def _RxN(self):
        if self._RxN_data is None:
            Q = self._cov["QS"][0][0]
            self._RxN_data = empty_like(Q.T)
        return self._RxN_data

    @property
    def _RxR(self):
        if self._RxR_data is None:
            Q = self._cov["QS"][0][0]
            r = Q.shape[1]
            self._RxR_data = empty((r, r))
        return self._RxR_data

    def _flush_cache(self):
        self._L_cache = None
        self._LQt_cache = None
        self._QS_cache = None
        self._AQ_cache = None
        self._TAQ_cache = None
        self._QSQtATQLQtA_cache = None

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

        Q = self._cov["QS"][0][0]
        S = self._cov["QS"][1]

        if S.size > 0:
            self.tau[:] = 1 / npsum((Q * sqrt(S)) ** 2, axis=1)
        else:
            self.tau[:] = 0.0
        self.eta[:] = self._mean
        self.eta[:] *= self.tau

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, v):
        self._initialize()
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
        from numpy_sugar.linalg import ddot, sum2diag
        from scipy.linalg import cho_factor

        if self._L_cache is not None:
            return self._L_cache

        Q = self._cov["QS"][0][0]
        S = self._cov["QS"][1]
        B = dot(Q.T, ddot(self._site.tau, Q, left=True))
        sum2diag(B, 1.0 / S, out=B)
        self._L_cache = cho_factor(B, lower=True)[0]
        return self._L_cache

    def LQt(self):
        from numpy_sugar.linalg import cho_solve

        if self._LQt_cache is not None:
            return self._LQt_cache

        L = self.L()
        Q = self._cov["QS"][0][0]

        self._LQt_cache = cho_solve(L, Q.T)
        return self._LQt_cache

    def QS(self):
        from numpy_sugar.linalg import ddot

        if self._QS_cache is not None:
            return self._QS_cache

        Q = self._cov["QS"][0][0]
        S = self._cov["QS"][1]

        self._QS_cache = ddot(Q, S)
        return self._QS_cache

    def update(self):
        from numpy_sugar.linalg import ddot, dotd

        self._flush_cache()

        Q = self._cov["QS"][0][0]

        K = dot(self.QS(), Q.T)

        BiQt = self.LQt()
        TK = ddot(self._site.tau, K, left=True)
        BiQtTK = dot(BiQt, TK)

        self.tau[:] = K.diagonal()
        self.tau -= dotd(Q, BiQtTK)
        self.tau[:] = 1 / self.tau

        if not all(self.tau >= 0.0):
            raise RuntimeError("'tau' has to be non-negative.")

        self.eta[:] = dot(K, self._site.eta)
        self.eta[:] += self._mean
        self.eta[:] -= dot(Q, dot(BiQtTK, self._site.eta))
        self.eta[:] -= dot(Q, dot(BiQt, self._site.tau * self._mean))

        self.eta *= self.tau
