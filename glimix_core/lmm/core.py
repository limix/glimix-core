from __future__ import division

from numpy import dot, log, maximum, sqrt, zeros
from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, economic_svd, rsolve, solve

from .scan import FastScanner


class LMMCore(object):
    def __init__(self, y, X, QS):
        self._QS = QS
        self._y = y

        self._tM = None
        self.__tbeta = None

        self._svd = None
        self.X = X

    def get_fast_scanner(self):
        return FastScanner(self._y, self.X, self._QS, self.delta)

    @property
    def X(self):
        # m = self.m
        return dot(self._svd[0], ddot(self._svd[1], self._svd[2], left=True))

    @X.setter
    def X(self, X):
        self._svd = economic_svd(X)
        self._tM = ddot(self._svd[0], sqrt(self._svd[1]), left=False)
        self.__tbeta = zeros(self._tM.shape[1])

    @property
    def m(self):
        r"""Returns :math:`\mathbf m = \mathrm X \boldsymbol\beta`."""
        return dot(self._tM, self._tbeta)

    @property
    def delta(self):
        raise NotImplementedError

    @delta.setter
    def delta(self, _):
        raise NotImplementedError

    @property
    def _tbeta(self):
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        self.__tbeta[:] = value

    @property
    def beta(self):
        SVs = ddot(self._svd[0], sqrt(self._svd[1]), left=False)
        z = rsolve(SVs, self.m)
        VsD = ddot(sqrt(self._svd[1]), self._svd[2], left=True)
        return rsolve(VsD, z)

    @beta.setter
    def beta(self, value):
        self._tbeta = sqrt(self._svd[1]) * dot(self._svd[2].T, value)

    @property
    def scale(self):
        a = [self._a(i) for i in [0, 1]]
        b = [self._b(i) for i in [0, 1]]
        c = [self._c(i) for i in [0, 1]]
        be = self.__tbeta
        p = [a[i] - 2 * b[i].dot(be) + be.dot(c[i]).dot(be) for i in [0, 1]]
        return maximum(sum(p) / len(self._y), epsilon.tiny)

    def _diag(self, i):
        if i == 0:
            return self._QS[1] * (1 - self.delta) + self.delta
        assert i == 1
        return self.delta

    def _a(self, i):
        return sum(self._yTQ_2x(i) / self._diag(i))

    def _b(self, i):
        return (self._yTQ(i) / self._diag(i)).dot(self._tMTQ(i).T)

    def _c(self, i):
        return (self._tMTQ(i) / self._diag(i)).dot(self._tMTQ(i).T)

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

    def update(self):
        self._update_fixed_effects()

    def lml(self):
        self.update()

        n = len(self._y)
        p = n - self._QS[1].shape[0]
        LOG2PI = 1.837877066409345339081937709124758839607238769531250
        lml = -n * LOG2PI - n - n * log(self.scale)
        lml += -sum(log(self._diag(0))) - p * log(self._diag(1))
        lml /= 2
        return lml
