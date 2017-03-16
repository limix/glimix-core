from __future__ import division

from copy import copy

from numpy import (ascontiguousarray, dot, log, sqrt, var, zeros)
from numpy.random import RandomState

from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, economic_svd, solve

from .scan import FastScanner


def _make_sure_has_variance(y):

    v = var(y)
    if v < epsilon.small:
        if v <= epsilon.tiny:
            random = RandomState(0)
            y = random.randn(len(y))
            y *= epsilon.small
        else:
            y = y / v
            y *= epsilon.small
    return y


class LMMCore(object):
    def __init__(self, y, M, QS):
        self._QS = QS
        self._y = _make_sure_has_variance(y)

        self._fix_scale = False

        self._n = self._y.shape[0]
        self._p = self._n - QS[1].shape[0]

        self._diag = [QS[1] * 0.5 + 0.5, 0.5]

        self._tM = None
        self.__tbeta = None
        self._covariate_setup(M)
        self._M = M

        d = M.shape[1]
        self._scale = 1.0
        self._delta = 0.5
        self._lml = 0.0

        self._a = [0.0, 0.0]

        self._b = [zeros(d), zeros(d)]
        self._c = [zeros((d, d)), zeros((d, d))]

        self._yTQ = [dot(self._y.T, QS[0][i]) for i in [0, 1]]

        self._yTQ_2x = [self._yTQ[i]**2 for i in [0, 1]]

        self._tMTQ = [self._tM.T.dot(QS[0][i]) for i in [0, 1]]

        self.valid_update = False
        # self.__QtymD = [None, None]

    def _covariate_setup(self, M):
        SVD = economic_svd(M)
        self._svd_U = SVD[0]
        self._svd_S12 = sqrt(SVD[1])
        self._svd_V = SVD[2]
        self._tM = ddot(self._svd_U, self._svd_S12, left=False)
        self.__tbeta = zeros(self._tM.shape[1])

    def get_fast_scanner(self):
        return FastScanner(self._y, self.M, self._QS, self.delta)

    def copy(self):
        # pylint: disable=W0212
        o = LMMCore.__new__(LMMCore)
        o._fix_scale = self._fix_scale
        o._n = self._n
        o._p = self._p
        o._diag = [copy(self._diag[i]) for i in [0, 1]]
        o._QS = self._QS
        o._M = self._M
        o._y = self._y

        o.__tbeta = self.__tbeta.copy()
        o._scale = self._scale
        o._delta = self._delta
        o._lml = self._lml

        o._a = [copy(self._a[i]) for i in [0, 1]]
        o._b = [copy(self._b[i]) for i in [0, 1]]
        o._c = [copy(self._c[i]) for i in [0, 1]]

        o._yTQ = self._yTQ
        o._yTQ_2x = self._yTQ_2x
        o._yTQ = self._yTQ
        o._tMTQ = self._tMTQ

        o.valid_update = self.valid_update

        return o

    def fix_scale(self):
        self._fix_scale = True

    def unfix_scale(self):
        self._fix_scale = False

    @property
    def M(self):
        return self._M

    @M.setter
    def M(self, v):
        self._M = v
        self._covariate_setup(v)
        d = self._tM.shape[1]
        self.__tbeta = zeros(d)

        self._b = [zeros(d), zeros(d)]
        self._c = [zeros((d, d)), zeros((d, d))]

        self._tMTQ = [self._tM.T.dot(self._QS[0][i]) for i in [0, 1]]

        self.valid_update = 0
        # self.__Q0tymD0 = None
        # self.__Q1tymD1 = None

    @property
    def m(self):
        r"""Returns :math:`\mathbf m = \mathrm M \boldsymbol\beta`."""
        return dot(self._tM, self._tbeta)

    # def _QtymD(self, i):
    #     if self.__QtymD[i] is None:
    #         Qtym[i] = self._yTQ[i] - self.__tbeta.dot(self._tMTQ[i])
    #         self.__QtymD[i] = Qtym[ii] / self._diag[i]
    #     return self.__QtymD[i]

    # def _Q0tymD0(self):
    #     if self.__Q0tymD0 is None:
    #         Q0tym = self._yTQ0 - self.__tbeta.dot(self._tMTQ0)
    #         self.__Q0tymD0 = Q0tym / self._diag0
    #     return self.__Q0tymD0
    #
    # def _Q1tymD1(self):
    #     if self.__Q1tymD1 is None:
    #         Q1tym = self._yTQ1 - self.__tbeta.dot(self._tMTQ1)
    #         self.__Q1tymD1 = Q1tym / self._diag1
    #     return self.__Q1tymD1

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
        self.valid_update = False
        # self.__QtymD = [None, None]
        self._delta = delta

    def _update_joints(self):
        for i in [0, 1]:
            self._a[i] = sum(self._yTQ_2x[i] / self._diag[i])
            self._b[i][:] = (self._yTQ[i] / self._diag[i]).dot(self._tMTQ[i].T)
            self._c[i][:] = (self._tMTQ[i] / self._diag[i]).dot(self._tMTQ[i].T)


    def _update_fixed_effects(self):
        nominator = self._b[1] - self._b[0]
        denominator = self._c[1] - self._c[0]
        self._tbeta = solve(denominator, nominator)

    def _update_scale(self):
        if self._fix_scale:
            return
        a = self._a
        b = self.__tbeta
        c = self._c
        p = [a[i] - 2 * self._b[i].dot(b) + b.dot(c[i]).dot(b) for i in [0, 1]]
        self._scale = sum(p) / self._n

    def _update_diags(self):
        self._diag[0][:] = self._QS[1]
        self._diag[0] *= (1 - self._delta)
        self._diag[0] += self._delta
        self._diag[1] = self._delta

    def update(self):
        if self.valid_update:
            return

        self._update_diags()
        self._update_joints()
        self._update_fixed_effects()
        self._update_scale()

        self.valid_update = True

    def lml(self):
        if self.valid_update:
            return self._lml

        self.update()

        n = self._n
        p = self._p
        LOG2PI = 1.837877066409345339081937709124758839607238769531250
        self._lml = -n * LOG2PI - n - n * log(self._scale)
        self._lml += -sum(log(self._diag[0])) - p * log(self._diag[1])
        self._lml /= 2
        return self._lml

    # def predict(self, covariates, Cp, Cpp):
    #     delta = self.delta
    #
    #     diag0 = self._diag0
    #     diag1 = self._diag1
    #
    #     CpQ0 = Cp.dot(self._QS[0][0])
    #     CpQ1 = Cp.dot(self._QS[0][1])
    #
    #     m = covariates.dot(self.beta)
    #     mean = m + (1 - delta) * CpQ0.dot(self._Q0tymD0())
    #     mean += (1 - delta) * CpQ1.dot(self._Q1tymD1())
    #
    #     cov = sum2diag(Cpp * (1 - self.delta), self.delta)
    #     cov -= (1 - delta)**2 * CpQ0.dot((CpQ0 / diag0).T)
    #     cov -= (1 - delta)**2 * CpQ1.dot((CpQ1 / diag1).T)
    #     cov *= self.scale
    #
    #     return LMMPredictor(mean, cov)


# class LMMPredictor(object):
#     def __init__(self, mean, cov):
#         self._mean = mean
#         self._cov = cov
#         self._mvn = multivariate_normal(mean, cov)
#
#     @property
#     def mean(self):
#         return self._mean
#
#     @property
#     def cov(self):
#         return self._cov
#
#     def pdf(self, y):
#         return self._mvn.pdf(y)
#
#     def logpdf(self, y):
#         return self._mvn.logpdf(y)
