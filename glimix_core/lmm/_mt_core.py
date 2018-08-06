from __future__ import division

from numpy import all as npall
from numpy import (
    asarray,
    atleast_2d,
    clip,
    dot,
    errstate,
    exp,
    full,
    isfinite,
    log,
    maximum,
    sqrt,
)
from numpy import sum as npsum
from numpy import zeros

from glimix_core.util import log2pi
from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, economic_svd, rsolve, sum2diag

from optimix import Function, Scalar

from ..util import economic_qs_zeros, numbers


class MTLMMCore(Function):
    def __init__(self, y, X=None, QS=None, SVD=None):
        Function.__init__(self, logistic=Scalar(0.0))

        y = [asarray(yi, float).ravel() for yi in y]

        if X is not None:
            X = [atleast_2d(asarray(Xi, float).T).T for Xi in X]
            if not all([npall(isfinite(Xi)) for Xi in X]):
                msg = "Not all values are finite in the covariates matrix."
                raise ValueError(msg)
            n = X[0].shape[0]
        else:
            n = SVD[0][0].shape[0]

        if not all([npall(isfinite(yi)) for yi in y]):
            raise ValueError("Not all values are finite in the outcome array.")

        if QS is None:
            QS = economic_qs_zeros(n)
            self.delta = 1.0
            super(MTLMMCore, self).fix("logistic")
        else:
            self.delta = 0.5

        if not isinstance(QS, tuple):
            raise ValueError("I was expecting a tuple for the covariance ")

        if QS[0][0].shape[0] != y[0].shape[0]:
            raise ValueError(
                "Number of samples differs between outcome"
                " and covariance decomposition"
                "decomposition"
            )

        if any([yi.shape[0] != n for yi in y]):
            raise ValueError(
                "Number of samples differs between outcome " "and covariates."
            )

        self.variables().get("logistic").bounds = (-numbers.logmax, +numbers.logmax)

        self._QS = QS
        self._y = y
        self._scale = 1.0
        self._fix_scale = False
        self._fix_beta = False

        self._tM = None
        self._tbeta = None

        self._svd = None
        self._set_X(X=X, SVD=SVD)
        self._verbose = False

    @property
    def _D(self):
        D = []
        n = self._y[0].shape[0]
        if self._QS[1].size > 0:
            D += [self._QS[1] * (1 - self.delta) + self.delta]
        if self._QS[1].size < n:
            D += [full(n - self._QS[1].size, self.delta)]
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
        lml -= sum(npsum(log(D)) for D in self._D)

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

        lml -= sum(npsum(log(D)) for D in self._D)

        d = (mTQ - yTQ for (mTQ, yTQ) in zip(self._mTQ, self._yTQ))
        lml += sum(dot(j / i, l) for (i, j, l) in zip(self._D, d, self._yTQ)) / s

        return lml / 2

    @property
    def _mTQDiQTm(self):
        return ((dot(i / j, i.T) for (i, j) in zip(ii, self._D)) for ii in self._tMTQ)

    @property
    def _mTQ(self):
        return (dot(self.mean.T, Q) for Q in self._QS[0] if Q.size > 0)

    def _optimal_scale(self):
        yTQDiQTy = self._yTQDiQTy
        yTQDiQTm = self._yTQDiQTm
        mTQDiQTm = self._mTQDiQTm
        b = self._tbeta
        p0 = sum(
            sum(i - 2 * dot(j, bi) for (i, j) in zip(ii, jj))
            for (ii, jj, bi) in zip(yTQDiQTy, yTQDiQTm, b)
        )
        p1 = sum(sum(dot(dot(bi, i), bi) for i in ii) for (ii, bi) in zip(mTQDiQTm, b))
        return maximum((p0 + p1) / sum(len(yi) for yi in self._y), epsilon.small)

    def _set_X(self, X=None, SVD=None):
        if SVD is None:
            self._svd = [economic_svd(Xi) for Xi in X]
        else:
            self._svd = SVD
        self._tM = [ddot(svd[0], sqrt(svd[1])) for svd in self._svd]
        self._tbeta = [zeros(tM.shape[1]) for tM in self._tM]

    @property
    def _tMTQ(self):
        return ((tM.T.dot(Q) for Q in self._QS[0] if Q.size > 0) for tM in self._tM)

    def _update_fixed_effects(self):
        if self.isfixed("beta"):
            return
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
        return ((dot(yi.T, Q) for Q in self._QS[0] if Q.size > 0) for yi in self._y)

    @property
    def _yTQQTy(self):
        return ((yTQi ** 2 for yTQi in yTQ) for yTQ in self._yTQ)

    @property
    def _yTQDiQTy(self):
        return ((npsum(i / j) for (i, j) in zip(ii, self._D)) for ii in self._yTQQTy)

    @property
    def _yTQDiQTm(self):
        yTQ = self._yTQ
        D = self._D
        tMTQ = self._tMTQ
        return (
            (dot(i / j, l.T) for (i, j, l) in zip(ii, D, ll))
            for (ii, ll) in zip(yTQ, tMTQ)
        )

    @property
    def X(self):
        r"""Covariates set by the user.

        It has to be a matrix of number-of-samples by number-of-covariates.

        Returns
        -------
        :class:`numpy.ndarray`
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
        :class:`numpy.ndarray`
            Optimal fixed-effect sizes.
        """
        SVs = ddot(self._svd[0], sqrt(self._svd[1]))
        z = rsolve(SVs, self.mean)
        VsD = ddot(sqrt(self._svd[1]), self._svd[2])
        return rsolve(VsD, z)

    @beta.setter
    def beta(self, beta):
        beta = asarray(beta, float)
        VsD = ddot(sqrt(self._svd[1]), self._svd[2])
        self._tbeta[:] = dot(VsD, beta)

    @property
    def delta(self):
        r"""Variance ratio between ``K`` and ``I``."""
        v = float(self.variables().get("logistic").value)
        with errstate(over="ignore", under="ignore"):
            v = 1 / (1 + exp(-v))
        return clip(v, epsilon.tiny, 1 - epsilon.tiny)

    @delta.setter
    def delta(self, delta):
        delta = clip(delta, epsilon.tiny, 1 - epsilon.tiny)
        self.variables().set(dict(logistic=log(delta / (1 - delta))))

    def lml(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        if self._verbose:
            print("Scale: {:g}".format(self.scale))
            print("Delta: {:g}".format(self.delta))
            print("Beta: {:}".format(str(self.beta)))
        if self.isfixed("scale"):
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
        if self.isfixed("scale"):
            return self._scale
        return self._optimal_scale()

    @scale.setter
    def scale(self, scale):
        self._scale = scale

    def mean_star(self, Xstar):
        return dot(Xstar, self.beta)

    def variance_star(self, kss):
        return kss * self.v0 + self.v1

    def covariance_star(self, ks):
        return ks * self.v0

    def covariance(self):
        r"""Covariance of the prior.

        Returns
        -------
        :class:`numpy.ndarray`
            :math:`v_0 \mathrm K + v_1 \mathrm I`.
        """
        Q0 = self._QS[0][0]
        S0 = self._QS[1]
        return sum2diag(dot(ddot(Q0, self.v0 * S0), Q0.T), self.v1)
