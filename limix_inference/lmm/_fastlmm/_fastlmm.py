from __future__ import division

from numpy import exp
from numpy import clip
from numpy import atleast_2d

from numpy_sugar import is_all_finite

from optimix import maximize_scalar
from optimix import Function
from optimix import Scalar

from ._core import FastLMMCore


class FastLMM(Function):
    r"""Fast Linear Mixed Models inference based on the covariance rank."""

    def __init__(self, y, Q0, Q1, S0, covariates=None):
        super(FastLMM, self).__init__(logistic=Scalar(0.0))

        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the phenotype.")

        self._flmmc = FastLMMCore(y, covariates, Q0, Q1, S0)
        self.set_nodata()

    @property
    def M(self):
        return self._flmmc.M

    @M.setter
    def M(self, v):
        self._flmmc.M = v

    def copy(self):
        o = FastLMM.__new__(FastLMM)
        super(FastLMM, o).__init__(logistic=Scalar(self.get('logistic')))
        o._flmmc = self._flmmc.copy()
        o.set_nodata()
        return o

    def _delta(self):
        v = clip(self.get('logistic'), -20, 20)
        x = 1 / (1 + exp(-v))
        return clip(x, 1e-5, 1 - 1e-5)

    @property
    def heritability(self):
        t = (self.fixed_effects_variance + self.genetic_variance +
             self.environmental_variance)
        return self.genetic_variance / t

    @property
    def fixed_effects_variance(self):
        return self._flmmc.m.var()

    @property
    def genetic_variance(self):
        return self._flmmc.scale * (1 - self._flmmc.delta)

    @property
    def environmental_variance(self):
        return self._flmmc.scale * self._flmmc.delta

    @property
    def beta(self):
        return self._flmmc.beta

    @property
    def m(self):
        return self._flmmc.m

    def learn(self, progress=None):
        maximize_scalar(self, progress=progress)
        self._flmmc.delta = self._delta()

    def value(self):
        self._flmmc.delta = self._delta()
        return self._flmmc.lml()

    def lml(self):
        self._flmmc.delta = self._delta()
        return self._flmmc.lml()

    def predict(self, X, covariates, Xp, trans=None):
        covariates = atleast_2d(covariates)
        Xp = atleast_2d(Xp)
        if trans is not None:
            Xp = trans.transform(Xp)
        Cp = Xp.dot(X.T)
        Cpp = Xp.dot(Xp.T)
        return self._flmmc.predict(covariates, Cp, Cpp)
