from __future__ import division

from numpy import exp
from numpy import clip
from numpy import atleast_2d

from numpy_sugar import is_all_finite

from optimix import maximize_scalar
from optimix import Function
from optimix import Scalar

from .core import FastLMMCore


class FastLMM(Function):
    r"""Fast Linear Mixed Models inference based on eigen decomposition.

    It models

    .. math::

        \mathbf y \sim \mathcal N\Big(~ \mathrm M\boldsymbol\beta;~
          s \big(
            (1-\delta)
              \mathrm Q \mathrm S \mathrm Q^{\intercal} +
            \delta \mathrm I
          \big)
        ~\Big)

    for which :math:`\mathrm Q \mathrm S \mathrm Q^{\intercal} = \mathrm K`
    is the eigen decomposition of the covariance matrix.

    Args:
        y (array_like): real-valued outcome.
        M (array_like): covariates as a two-dimensional array.
        QS (tuple): economic eigen decompositon ((Q0, Q1), S0).
        options (dict): nao sei

    Examples
    --------

    .. doctest::

        >>> from numpy import array
        >>> from numpy_sugar.linalg import economic_qs_linear
        >>> from limix_inference.lmm import FastLMM
        >>>
        >>> X = array([[1, 2], [3, -1]], float)
        >>> QS = economic_qs_linear(X)
        >>> covariates = array([[1], [1]])
        >>> y = array([-1, 2], float)
        >>> flmm = FastLMM(y, covariates, QS)
        >>> flmm.learn(progress=False)
        >>> print('%.3f' % flmm.lml())
        -3.649

    One can also specify which parameters will be fitted as follows.

    .. doctest::

        >>> from numpy import array
        >>> from numpy_sugar.linalg import economic_qs_linear
        >>> from limix_inference.lmm import FastLMM
        >>>
        >>> X = array([[1, 2], [3, -1]], float)
        >>> QS = economic_qs_linear(X)
        >>> covariates = array([[1], [1]])
        >>> y = array([-1, 2], float)
        >>> flmm = FastLMM(y, covariates, QS)
        >>> flmm.fix('delta')
        >>> flmm.fix('scale')
        >>> flmm.learn(progress=False)
        >>> print('%.3f' % flmm.lml())
        -4.232
        >>> print('%.1f' % flmm.heritability)
        0.5
        >>> flmm.unfix('delta')
        >>> flmm.learn(progress=False)
        >>> print('%.3f' % flmm.lml())
        -2.838
        >>> print('%.1f' % flmm.heritability)
        0.0

    """

    def __init__(self, y, M, QS, options=None):
        super(FastLMM, self).__init__(logistic=Scalar(0.0))
        self._options = options

        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the phenotype.")

        self._flmmc = FastLMMCore(y, M, QS)
        self.set_nodata()

    def fix(self, var_name):
        if var_name == 'delta':
            super(FastLMM, self).fix('logistic')
        elif var_name == 'scale':
            self._flmmc.fix_scale()
        else:
            raise ValueError

        self._flmmc.valid_update = False

    def unfix(self, var_name):
        if var_name == 'delta':
            super(FastLMM, self).unfix('logistic')
        elif var_name == 'scale':
            self._flmmc.unfix_scale()
        else:
            raise ValueError

        self._flmmc.valid_update = False

    def get_normal_likelihood_trick(self):
        return self._flmmc.get_normal_likelihood_trick()

    @property
    def M(self):
        return self._flmmc.M

    @M.setter
    def M(self, v):
        self._flmmc.M = v

    def copy(self):
        # pylint: disable=W0212
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

    def learn(self, progress=True):
        maximize_scalar(self, progress=progress)
        self._flmmc.update()
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
