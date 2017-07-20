from __future__ import division

from numpy import clip, dot, exp, log
from numpy_sugar import epsilon, is_all_finite
from optimix import Function, Scalar, maximize_scalar

from .core import LMMCore


class LMM(LMMCore, Function):
    r"""Fast Linear Mixed Models inference via maximum likelihood.

    It models

    .. math::

        \mathbf y \sim \mathcal N\Big(~ \mathrm X\boldsymbol\beta;~
          s \big(
            (1-\delta)
              \mathrm Q \mathrm S \mathrm Q^{\intercal} +
            \delta \mathrm I
          \big)
        ~\Big)

    for which :math:`\mathrm Q\mathrm S\mathrm Q^{\intercal}=\tilde{\mathrm K}`
    is the eigen decomposition of :math:`\tilde{\mathrm K}`.

    Args:
        y (array_like): outcome.
        X (array_like): covariates as a two-dimensional array.
        QS (tuple): economic eigen decompositon ((Q0, Q1), S0).

    Examples
    --------

    .. doctest::

        >>> from numpy import array
        >>> from numpy_sugar.linalg import economic_qs_linear
        >>> from glimix_core.lmm import LMM
        >>>
        >>> X = array([[1, 2], [3, -1]], float)
        >>> QS = economic_qs_linear(X)
        >>> covariates = array([[1], [1]])
        >>> y = array([-1, 2], float)
        >>> lmm = LMM(y, covariates, QS)
        >>> lmm.learn(verbose=False)
        >>> print('%.3f' % lmm.lml())
        -3.649

    One can also specify which parameters should be fitted:

    .. doctest::

        >>> from numpy import array
        >>> from numpy_sugar.linalg import economic_qs_linear
        >>> from glimix_core.lmm import LMM
        >>>
        >>> X = array([[1, 2], [3, -1]], float)
        >>> QS = economic_qs_linear(X)
        >>> covariates = array([[1], [1]])
        >>> y = array([-1, 2], float)
        >>> lmm = LMM(y, covariates, QS)
        >>> lmm.fix('delta')
        >>> lmm.fix('scale')
        >>> lmm.delta = 0.5
        >>> lmm.scale = 1
        >>> lmm.learn(verbose=False)
        >>> print('%.3f' % lmm.lml())
        -4.232
        >>> print('%.1f' % lmm.heritability)
        0.5
        >>> lmm.unfix('delta')
        >>> lmm.learn(verbose=False)
        >>> print('%.3f' % lmm.lml())
        -2.838
        >>> print('%.1f' % lmm.heritability)
        0.0
    """

    def __init__(self, y, X, QS):
        LMMCore.__init__(self, y, X, QS)
        Function.__init__(self, logistic=Scalar(0.0))

        if not is_all_finite(y):
            raise ValueError("There are non-finite values in the phenotype.")

        self._fix = dict(beta=False, scale=False)
        self._delta = 0.5
        self._scale = LMMCore.scale.fget(self)  # pylint: disable=E1101
        self.set_nodata()

    def fix(self, var_name):
        if var_name == 'delta':
            super(LMM, self).fix('logistic')
        else:
            self._fix[var_name] = True

    def unfix(self, var_name):
        if var_name == 'delta':
            super(LMM, self).unfix('logistic')
        else:
            self._fix[var_name] = False

    @property
    def scale(self):
        if self._fix['scale']:
            return self._scale
        return LMMCore.scale.fget(self)  # pylint: disable=E1101

    @scale.setter
    def scale(self, v):
        self._scale = v

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        delta = clip(delta, epsilon.tiny, 1 - epsilon.tiny)
        self._delta = delta
        self.variables().set(dict(logistic=log(delta / (1 - delta))))

    def copy(self):
        # pylint: disable=W0212
        o = LMM.__new__(LMM)

        LMMCore.__init__(o, self._y, self.X, self._QS)
        Function.__init__(
            o, logistic=Scalar(self.variables().get('logistic').value))

        o._fix = self._fix.copy()
        o._delta = self._delta
        o._scale = self._scale

        o.set_nodata()
        return o

    def _get_delta(self):
        v = clip(self.variables().get('logistic').value, -20, 20)
        x = 1 / (1 + exp(-v))
        return clip(x, 1e-5, 1 - 1e-5)

    @property
    def heritability(self):
        t = (self.fixed_effects_variance + self.genetic_variance +
             self.environmental_variance)
        return self.genetic_variance / t

    @property
    def fixed_effects_variance(self):
        return self.m.var()

    @property
    def genetic_variance(self):
        return self.scale * (1 - self.delta)

    @property
    def environmental_variance(self):
        return self.scale * self.delta

    def learn(self, verbose=True):
        maximize_scalar(self, verbose=verbose)
        self.update()
        self.delta = self._get_delta()

    def value(self):
        self.delta = self._get_delta()
        return self.lml()

    def lml(self):
        self.delta = self._get_delta()
        return LMMCore.lml(self)

    def mean(self):
        return self.m
