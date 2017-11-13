from __future__ import division

from numpy import sum as npsum
from numpy import asarray, atleast_2d, dot, log, maximum, sqrt, zeros, isfinite
from numpy import all as npy_all

from glimix_core.util import log2pi
from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, economic_svd, rsolve


class LMMCore(object):
    def __init__(self, y, X, QS):
        y = asarray(y, float).ravel()
        X = atleast_2d(asarray(X, float).T).T
        if not npy_all(isfinite(X)):
            raise ValueError("Not all values are finite in the covariates matrix.")

        if not npy_all(isfinite(y)):
            raise ValueError(
                "Not all values are finite in the outcome array.")

        if not isinstance(QS, tuple):
            raise ValueError("I was expecting a tuple for the covariance "
                             "decomposition")

        if y.shape[0] != X.shape[0]:
            raise ValueError("Number of samples differs between outcome "
                             "and covariates.")

        if QS[0][0].shape[0] != y.shape[0]:
            raise ValueError("Number of samples differs between outcome"
                             " and covariance decomposition")

        self._QS = QS
        self._y = y
        self._scale = 1.0
        self._fix_scale = False

        self._tM = None
        self._tbeta = None

        self._svd = None
        self._set_X(X)

    @property
    def _D(self):
        D = [self._QS[1] * (1 - self.delta) + self.delta]
        if self._QS[0][1].size > 0:
            D += [self.delta]
        return D

    def _lml_optimal_scale(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.

        """
        self._update()

        n = len(self._y)
        lml = -n * log2pi - n - n * log(self.scale)
        lml -= npsum(log(self._D[0]))
        if n > self._QS[1].shape[0]:
            lml -= (n - self._QS[1].shape[0]) * log(self._D[1])

        return lml / 2

    def _lml_arbitrary_scale(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        self._update()
        s = self.scale

        n = len(self._y)
        lml = -n * log2pi - n * log(s)

        lml -= npsum(log(self._D[0]))
        if n > self._QS[1].shape[0]:
            lml -= (n - self._QS[1].shape[0]) * log(self._D[1])

        d = (mTQ - yTQ for (mTQ, yTQ) in zip(self._mTQ, self._yTQ))
        lml += sum(
            dot(j / i, l) for (i, j, l) in zip(self._D, d, self._yTQ)) / s

        return lml / 2

    @property
    def _mTQDiQTm(self):
        return (dot(i / j, i.T) for (i, j) in zip(self._tMTQ, self._D))

    @property
    def _mTQ(self):
        return (dot(self.mean.T, Q) for Q in self._QS[0] if Q.size > 0)

    def _optimal_scale(self):
        yTQDiQTy = self._yTQDiQTy
        yTQDiQTm = self._yTQDiQTm
        b = self._tbeta
        p0 = sum(i - 2 * dot(j, b) for (i, j) in zip(yTQDiQTy, yTQDiQTm))
        p1 = sum(dot(dot(b, i), b) for i in self._mTQDiQTm)
        return maximum((p0 + p1) / len(self._y), epsilon.tiny)

    def _set_X(self, X):
        self._svd = economic_svd(X)
        self._tM = ddot(self._svd[0], sqrt(self._svd[1]), left=False)
        self._tbeta = zeros(self._tM.shape[1])

    @property
    def _tMTQ(self):
        return (self._tM.T.dot(Q) for Q in self._QS[0] if Q.size > 0)

    def _update_fixed_effects(self):
        yTQDiQTm = list(self._yTQDiQTm)
        mTQDiQTm = list(self._mTQDiQTm)
        nominator = yTQDiQTm[0]
        denominator = mTQDiQTm[0]

        if len(yTQDiQTm) > 1:
            nominator += yTQDiQTm[1]
            denominator += mTQDiQTm[1]

        self._tbeta[:] = rsolve(denominator, nominator)

    def _update(self):
        self._update_fixed_effects()

    @property
    def _yTQ(self):
        return (dot(self._y.T, Q) for Q in self._QS[0] if Q.size > 0)

    @property
    def _yTQQTy(self):
        return (yTQ**2 for yTQ in self._yTQ)

    @property
    def _yTQDiQTy(self):
        return (npsum(i / j) for (i, j) in zip(self._yTQQTy, self._D))

    @property
    def _yTQDiQTm(self):
        yTQ = self._yTQ
        D = self._D
        tMTQ = self._tMTQ
        return (dot(i / j, l.T) for (i, j, l) in zip(yTQ, D, tMTQ))

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

    @property
    def delta(self):
        raise NotImplementedError

    @delta.setter
    def delta(self, _):
        raise NotImplementedError

    def isfixed(self, var_name):
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
