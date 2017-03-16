from __future__ import division

from copy import copy

from numpy import ascontiguousarray, dot, log, sqrt, var, zeros
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

        self._diag = [QS[1] * 0.5 + 0.5, 0.5]

        self._tM = None
        self.__tbeta = None
        # self._covariate_setup(M)

        self._svd = None
        self.M = M
        # self._svd = economic_svd(M)
        # self._tM = ddot(self._svd[0], sqrt(self._svd[1]), left=False)
        # self.__tbeta = zeros(self._tM.shape[1])

        # self._M = M

        self._scale = 1.0
        self._delta = 0.5

        # self._yTQ = [dot(self._y.T, QS[0][i]) for i in [0, 1]]
        # self._yTQ_2x = [self._yTQ[i]**2 for i in [0, 1]]
        # self._tMTQ = [self._tM.T.dot(QS[0][i]) for i in [0, 1]]

    # def _covariate_setup(self, M):
    #     SVD = economic_svd(M)
    #     self._svd_U = SVD[0]
    #     self._svd_S12 = sqrt(SVD[1])
    #     self._svd_V = SVD[2]
    #     self._tM = ddot(self._svd_U, self._svd_S12, left=False)
    #     self.__tbeta = zeros(self._tM.shape[1])

    def get_fast_scanner(self):
        return FastScanner(self._y, self.M, self._QS, self.delta)

    def copy(self):
        # pylint: disable=W0212
        o = LMMCore.__new__(LMMCore)
        o._fix_scale = self._fix_scale
        o._diag = [copy(self._diag[i]) for i in [0, 1]]
        o._QS = self._QS
        # o._M = self._M
        o._y = self._y

        o.__tbeta = self.__tbeta.copy()
        o._scale = self._scale
        o._delta = self._delta

        # o._yTQ = self._yTQ
        # o._yTQ_2x = self._yTQ_2x
        # o._tMTQ = self._tMTQ

        return o

    def fix_scale(self):
        self._fix_scale = True

    def unfix_scale(self):
        self._fix_scale = False

    @property
    def M(self):
        return dot(self._svd[0], ddot(self._svd[1], self._svd[2], left=True))

    @M.setter
    def M(self, M):
        self._svd = economic_svd(M)
        self._tM = ddot(self._svd[0], sqrt(self._svd[1]), left=False)
        self.__tbeta = zeros(self._tM.shape[1])
        # self._M = v
        # self._covariate_setup(v)
        # d = self._tM.shape[1]
        # self.__tbeta = zeros(d)
        # self._tMTQ = [self._tM.T.dot(self._QS[0][i]) for i in [0, 1]]

    @property
    def m(self):
        r"""Returns :math:`\mathbf m = \mathrm M \boldsymbol\beta`."""
        return dot(self._tM, self._tbeta)

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
        return solve(self._svd[2].T, self._tbeta / sqrt(self._svd[1]))

    @beta.setter
    def beta(self, value):
        self._tbeta = sqrt(self._svd[1]) * dot(self._svd[2].T, value)

    @property
    def scale(self):
        return self._scale

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        self._delta = delta

    def _a(self, i):
        return sum(self._yTQ_2x(i) / self._diag[i])

    def _b(self, i):
        return (self._yTQ(i) / self._diag[i]).dot(self._tMTQ(i).T)

    def _c(self, i):
        return (self._tMTQ(i) / self._diag[i]).dot(self._tMTQ(i).T)

    def _yTQ(self, i):
        return dot(self._y.T, self._QS[0][i])

    def _yTQ_2x(self, i):
        return self._yTQ(i)**2

    def _tMTQ(self, i):
        return self._tM.T.dot(self._QS[0][i])

    def _update_fixed_effects(self):
        nominator = self._b(1) - self._b(0)
        denominator = self._c(1) - self._c(0)
        self._tbeta = solve(denominator, nominator)

    def _update_scale(self):
        if self._fix_scale:
            return
        a = [self._a(i) for i in [0, 1]]
        b = [self._b(i) for i in [0, 1]]
        c = [self._c(i) for i in [0, 1]]
        be = self.__tbeta
        p = [a[i] - 2 * b[i].dot(be) + be.dot(c[i]).dot(be) for i in [0, 1]]
        self._scale = sum(p) / len(self._y)

    def _update_diags(self):
        self._diag[0][:] = self._QS[1]
        self._diag[0] *= (1 - self._delta)
        self._diag[0] += self._delta
        self._diag[1] = self._delta

    def update(self):
        self._update_diags()
        self._update_fixed_effects()
        self._update_scale()

    def lml(self):
        self.update()

        n = len(self._y)
        p = n - self._QS[1].shape[0]
        LOG2PI = 1.837877066409345339081937709124758839607238769531250
        lml = -n * LOG2PI - n - n * log(self._scale)
        lml += -sum(log(self._diag[0])) - p * log(self._diag[1])
        lml /= 2
        return lml
