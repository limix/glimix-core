from __future__ import division

from numpy import sum as npsum
from numpy import dot, log, maximum, sqrt, zeros
from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, economic_svd, rsolve, solve

from ..util import log2pi


class LMMCore(object):
    def __init__(self, y, X, QS):
        self._QS = QS
        self._y = y
        self._scale = 1.0
        self._fix_scale = False

        self._tM = None
        self.__tbeta = None

        self._svd = None
        self._set_X(X)

    def _diag(self, i):
        if i == 0:
            return self._QS[1] * (1 - self.delta) + self.delta
        return self.delta

    @property
    def _D(self):
        return (self._QS[1] * (1 - self.delta) + self.delta, self.delta)

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

    def _lml_optimal_scale(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        self._update()

        n = len(self._y)
        p = n - self._QS[1].shape[0]
        lml = -n * log2pi - n - n * log(self.scale)
        lml += -sum(log(self._diag(0))) - p * log(self._diag(1))
        lml /= 2
        return lml

    def _lml_arbitrary_scale(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        self._update()

        n = len(self._y)
        p = n - self._QS[1].shape[0]

        lml = -n * log2pi
        lml -= n * log(self.scale)
        lml -= npsum(log(self._diag(0))) + p * log(self._diag(1))
        m = self.mean - self._y
        m = [dot(m, Q) for Q in self._QS[0] if Q.size > 0]
        D = self._D
        lml += npsum(
            dot(D[i] * m[i], self._yTQ(i)) for i in range(2)
            if self._yTQ(i).size > 0)

        return lml / 2

    def _set_X(self, X):
        self._svd = economic_svd(X)
        self._tM = ddot(self._svd[0], sqrt(self._svd[1]), left=False)
        self.__tbeta = zeros(self._tM.shape[1])

    def _tMTQ(self, i):
        return self._tM.T.dot(self._QS[0][i])

    def _update_fixed_effects(self):
        nominator = self._b(1) - self._b(0)
        denominator = self._c(1) - self._c(0)
        self._tbeta = solve(denominator, nominator)

    def _update(self):
        self._update_fixed_effects()

    def _optimal_scale(self):
        a = [self._a(i) for i in [0, 1]]
        b = [self._b(i) for i in [0, 1]]
        c = [self._c(i) for i in [0, 1]]
        be = self.__tbeta
        p = [a[i] - 2 * b[i].dot(be) + be.dot(c[i]).dot(be) for i in [0, 1]]
        return maximum(sum(p) / len(self._y), epsilon.tiny)

    @property
    def _tbeta(self):
        return self.__tbeta

    @_tbeta.setter
    def _tbeta(self, value):
        self.__tbeta[:] = value

    @property
    def X(self):
        r"""Covariates set by the user.

        It has to be a matrix of number-of-samples by number-of-covariates.

        Returns
        -------
        array_like
            Covariates.
        """
        return dot(self._svd[0], ddot(self._svd[1], self._svd[2], left=True))

    @X.setter
    def X(self, X):
        self._set_X(X)

    def isfixed(self, var_name):
        raise NotImplementedError

    @property
    def mean(self):
        r"""Mean of the prior.

        Formally, :math:`\mathbf m = \mathrm X \boldsymbol\beta`.

        Returns
        -------
        array_like
            Mean of the prior.
        """
        return dot(self._tM, self._tbeta)

    @property
    def beta(self):
        r"""Fixed-effect sizes.

        The optimal fixed-effect sizes is given by any solution to equation

        .. math::

            (\mathrm Q^{\intercal}\mathrm X)^{\intercal}
                \mathrm D^{-1}
                (\mathrm Q^{\intercal}\mathrm X)
                \boldsymbol\beta =
                (\mathrm Q^{\intercal}\mathrm X)^{\intercal}
                \mathrm D^{-1}
                (\mathrm Q^{\intercal}\mathbf y).

        Returns
        -------
        array_like
            Optimal fixed-effect sizes.
        """
        SVs = ddot(self._svd[0], sqrt(self._svd[1]), left=False)
        z = rsolve(SVs, self.mean)
        VsD = ddot(sqrt(self._svd[1]), self._svd[2], left=True)
        return rsolve(VsD, z)

    # @beta.setter
    # def beta(self, value):
    #     self._tbeta = sqrt(self._svd[1]) * dot(self._svd[2].T, value)

    @property
    def delta(self):
        raise NotImplementedError

    @delta.setter
    def delta(self, _):
        raise NotImplementedError

    def lml(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        if self.isfixed('scale'):
            return self._lml_arbitrary_scale()
        return self._lml_optimal_scale()

    @property
    def scale(self):
        r"""Scaling factor.

        Returns
        -------
        float
            Current scale if fixed; optimal scale otherwise.
        """
        if self.isfixed('scale'):
            return self._scale
        return self._optimal_scale()

    @scale.setter
    def scale(self, scale):
        self._scale = scale
