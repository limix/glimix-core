from __future__ import division

from scipy.stats import multivariate_normal
from numpy import (dot, log, var, zeros, sqrt, ascontiguousarray)

from numpy_sugar.linalg import (sum2diag, solve, economic_svd, ddot)


class FastLMMCore(object):
    def __init__(self, y, M, Q0, Q1, S0):
        if var(y) < 1e-8:
            raise ValueError("The phenotype variance is too low: %e." % var(y))

        self._n = y.shape[0]
        self._p = self._n - S0.shape[0]
        self._S0 = S0
        self._diag0 = S0 * 0.5 + 0.5
        self._diag1 = 0.5
        self._Q0 = Q0
        self._Q1 = Q1
        self._tM = None
        self.__tbeta = None
        self._covariate_setup(M)
        self._M = M

        d = M.shape[1]
        self._scale = 1.0
        self._delta = 0.5
        self._lml = 0.0

        self._a1 = 0.0
        self._b1 = zeros(d)
        self._c1 = zeros((d, d))

        self._a0 = 0.0
        self._b0 = zeros(d)
        self._c0 = zeros((d, d))

        self._yTQ0 = dot(y.T, Q0)
        self._yTQ0_2x = self._yTQ0**2

        self._yTQ1 = dot(y.T, Q1)
        self._yTQ1_2x = self._yTQ1**2

        self._oneTQ0 = self._tM.T.dot(Q0)
        self._oneTQ1 = self._tM.T.dot(Q1)

        self._valid_update = 0
        self.__Q0tymD0 = None
        self.__Q1tymD1 = None

    def _covariate_setup(self, M):
        SVD = economic_svd(M)
        self._svd_U = SVD[0]
        self._svd_S12 = sqrt(SVD[1])
        self._svd_V = SVD[2]
        self._tM = ddot(self._svd_U, self._svd_S12, left=False)

    def copy(self):
        o = FastLMMCore.__new__(FastLMMCore)
        o._n = self._n
        o._p = self._p
        o._S0 = self._S0
        o._diag0 = self._diag0.copy()
        o._diag1 = self._diag1
        o._Q0 = self._Q0
        o._Q1 = self._Q1
        o._M = self._M

        o.__tbeta = self.__tbeta.copy()
        o._scale = self._scale
        o._delta = self._delta
        o._lml = self._lml

        o._a1 = self._a1
        o._b1 = self._b1.copy()
        o._c1 = self._c1.copy()

        o._a0 = self._a0
        o._b0 = self._b0.copy()
        o._c0 = self._c0.copy()

        o._yTQ0 = self._yTQ0
        o._yTQ0_2x = self._yTQ0_2x

        o._yTQ1 = self._yTQ1
        o._yTQ1_2x = self._yTQ1_2x

        o._oneTQ0 = self._oneTQ0
        o._oneTQ1 = self._oneTQ1

        o._valid_update = self._valid_update
        from copy import copy
        o.__Q0tymD0 = copy(self.__Q0tymD0)
        o.__Q1tymD1 = copy(self.__Q1tymD1)

        return o

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, v):
        self._M = v
        self._covariate_setup(v)
        d = self._tM.shape[1]
        self.__tbeta = zeros(d)

        self._b1 = zeros(d)
        self._c1 = zeros((d, d))

        self._b0 = zeros(d)
        self._c0 = zeros((d, d))

        self._oneTQ0 = self._tM.T.dot(self._Q0)
        self._oneTQ1 = self._tM.T.dot(self._Q1)

        self._valid_update = 0
        self.__Q0tymD0 = None
        self.__Q1tymD1 = None

    @property
    def m(self):
        r"""Returns :math:`\mathbf m = \mathrm M \boldsymbol\beta`."""
        return dot(self._tM, self._tbeta)

    def _Q0tymD0(self):
        if self.__Q0tymD0 is None:
            Q0tym = self._yTQ0 - self.__tbeta.dot(self._oneTQ0)
            self.__Q0tymD0 = Q0tym / self._diag0
        return self.__Q0tymD0

    def _Q1tymD1(self):
        if self.__Q1tymD1 is None:
            Q1tym = self._yTQ1 - self.__tbeta.dot(self._oneTQ1)
            self.__Q1tymD1 = Q1tym / self._diag1
        return self.__Q1tymD1

    @property
    def _tbeta(self):
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        if self.__tbeta is None:
            self.__tbeta = ascontiguousarray(value, float).copy()
        else:
            self.__tbeta[:] = value

    @property
    def beta(self):
        r"""Returns :math:`\boldsymbol\beta`."""
        return solve(self._svd_V.T, self._tbeta / self._svd_S12)

    @beta.setter
    def beta(self, value):
        self._tbeta = self._svd_S12 * dot(self._svd_V.T, value)

    @property
    def scale(self):
        return self._scale

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._valid_update = 0
        self.__Q0tymD0 = None
        self.__Q1tymD1 = None
        self._delta = delta

    def _update_joints(self):
        yTQ0_2x = self._yTQ0_2x
        yTQ1_2x = self._yTQ1_2x

        self._a1 = yTQ1_2x.sum() / self._diag1
        self._b1[:] = (self._yTQ1 / self._diag1).dot(self._oneTQ1.T)
        self._c1[:] = (self._oneTQ1 / self._diag1).dot(self._oneTQ1.T)

        self._a0 = (yTQ0_2x / self._diag0).sum()
        self._b0[:] = (self._yTQ0 / self._diag0).dot(self._oneTQ0.T)
        self._c0[:] = (self._oneTQ0 / self._diag0).dot(self._oneTQ0.T)

    def _update_fixed_effects(self):
        nominator = self._b1 - self._b0
        denominator = self._c1 - self._c0
        self._tbeta = solve(denominator, nominator)

    def _update_scale(self):
        b = self.__tbeta
        p0 = self._a1 - 2 * self._b1.dot(b) + b.dot(self._c1.dot(b))
        p1 = self._a0 - 2 * self._b0.dot(b) + b.dot(self._c0).dot(b)
        self._scale = (p0 + p1) / self._n

    def _update_diags(self):
        self._diag0[:] = self._S0
        self._diag0 *= (1 - self._delta)
        self._diag0 += self._delta
        self._diag1 = self._delta

    def _update(self):
        if self._valid_update:
            return

        self._update_diags()
        self._update_joints()
        self._update_fixed_effects()
        self._update_scale()

        self._valid_update = 1

    def lml(self):
        if self._valid_update:
            return self._lml

        self._update()

        n = self._n
        p = self._p
        LOG2PI = 1.837877066409345339081937709124758839607238769531250
        self._lml = -n * LOG2PI - n - n * log(self._scale)
        self._lml += -sum(log(self._diag0)) - p * log(self._diag1)
        self._lml /= 2
        return self._lml

    def predict(self, covariates, Cp, Cpp):
        delta = self.delta

        diag0 = self._diag0
        diag1 = self._diag1

        CpQ0 = Cp.dot(self._Q0)
        CpQ1 = Cp.dot(self._Q1)

        m = covariates.dot(self.beta)
        mean = m + (1 - delta) * CpQ0.dot(self._Q0tymD0())
        mean += (1 - delta) * CpQ1.dot(self._Q1tymD1())

        cov = sum2diag(Cpp * (1 - self.delta), self.delta)
        cov -= (1 - delta)**2 * CpQ0.dot((CpQ0 / diag0).T)
        cov -= (1 - delta)**2 * CpQ1.dot((CpQ1 / diag1).T)
        cov *= self.scale

        return FastLMMPredictor(mean, cov)


class FastLMMPredictor(object):
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov
        self._mvn = multivariate_normal(mean, cov)

    @property
    def mean(self):
        return self._mean

    @property
    def cov(self):
        return self._cov

    def pdf(self, y):
        return self._mvn.pdf(y)

    def logpdf(self, y):
        return self._mvn.logpdf(y)
