from __future__ import division

from numpy import clip, exp, log
from numpy_sugar import epsilon, is_all_finite
from optimix import Function, Scalar, maximize_scalar

from .core import LMMCore
from .scan import FastScanner


class LMM(LMMCore, Function):
    r"""Fast Linear Mixed Models inference via maximum likelihood.

    It perform inference on the exactly the same model given at :eq:`lmm1`
    but with a different variance parameterization:

    .. math::
        :label: lmm2

        \mathbf y \sim \mathcal N\Big(~ \mathrm X\boldsymbol\beta;~
          s \big(
            (1-\delta)
              \mathrm K +
            \delta \mathrm I
          \big)
        ~\Big)


    for which ``K`` is the covariance matrix represented internally as an
    economic eigen decomposition.
    Naturally, we have :math:`v_0 = s (1 - \delta)` and
    :math:`v_1 = s \delta`.

    Implementation wise, we make use of the identity

    .. math::

        s^{-1}((1-\delta)\mathrm K + \delta\mathrm I)^{-1} = s^{-1}
            \mathrm Q \left(
            \begin{array}{cc}
                (1-\delta)\mathrm S_0 + \delta\mathrm I_0 & \mathbf 0\\
                \mathbf 0 & \delta\mathrm I_1
            \end{array}\right)^{-1}
            \mathrm Q^{\intercal},

    as it allows use to write the marginal likelihood as

    .. math::

        \mathcal N\left(\mathrm Q^{\intercal} \mathbf y ~|~
                   \mathrm Q^{\intercal} \mathbf m,~
                   s^{-1} \left(
                       \begin{array}{cc}
                           (1-\delta)\mathrm S_0 + \delta\mathrm I_0 &
                                \mathbf 0\\
                           \mathbf 0 & \delta\mathrm I_1
                       \end{array}\right)\right).


    Let us define

    .. math::

        \mathrm D = \left(
            \begin{array}{cc}
              (1-\delta)\mathrm S_0 + \delta\mathrm I_0 & \mathbf 0\\
              \mathbf 0 & \delta\mathrm I_1
            \end{array}
            \right),

    used later on in this page.


    Parameters
    ----------
    y : array_like
        Outcome.
    X : array_like
        Covariates as a two-dimensional array.
    QS : tuple
        Economic eigen decompositon in form of ``((Q0, Q1), S0)`` of a
        covariance matrix ``K``.

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
        self._scale = LMMCore.scale.fget(self)
        self.set_nodata()

    def _get_delta(self):
        v = clip(self.variables().get('logistic').value, -20, 20)
        x = 1 / (1 + exp(-v))
        return clip(x, 1e-5, 1 - 1e-5)

    @property
    def beta(self):
        return LMMCore.beta.fget(self)

    @beta.setter
    def beta(self, beta):
        LMMCore.beta.fset(self, beta)

    def copy(self):
        r"""Return a copy of this object."""
        o = LMM.__new__(LMM)

        LMMCore.__init__(o, self._y, self.X, self._QS)
        Function.__init__(
            o, logistic=Scalar(self.variables().get('logistic').value))

        o._fix = self._fix.copy()
        o._delta = self._delta
        o._scale = self._scale

        o.set_nodata()
        return o

    @property
    def delta(self):
        r"""Variance ratio between ``K`` and ``I``."""
        return self._delta

    @delta.setter
    def delta(self, delta):
        delta = clip(delta, epsilon.tiny, 1 - epsilon.tiny)
        self._delta = delta
        self.variables().set(dict(logistic=log(delta / (1 - delta))))

    @property
    def v0(self):
        r"""First variance.

        Returns
        -------
        float
            :math:`s (1 - \delta)`
        """
        return self.scale * (1 - self.delta)

    @property
    def v1(self):
        r"""Second variance.

        Returns
        -------
        float
            :math:`s \delta`
        """
        return self.scale * self.delta

    def fit(self, verbose=True):
        r"""Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        """
        maximize_scalar(self, 'LMM', verbose=verbose)
        self._update()
        self.delta = self._get_delta()

    def fix(self, var_name):
        r"""Disable the optimisation of a given variable.

        Parameters
        ----------
        var_name : str
            Possible values are `delta`, and TODO.
        """
        if var_name == 'delta':
            super(LMM, self).fix('logistic')
        else:
            self._fix[var_name] = True

    @property
    def fixed_effects_variance(self):
        return self.m.var()

    @property
    def genetic_variance(self):
        return self.scale * (1 - self.delta)

    def get_fast_scanner(self):
        r"""Return :class:`.FastScanner` for the current delta."""
        QS = (self._QS[0], (1 - self.delta) * self._QS[1])
        return FastScanner(self._y, self.X, QS, self.delta)

    def lml(self):
        self.delta = self._get_delta()
        return LMMCore.lml(self)

    @property
    def m(self):
        return super(LMM, self).m.fget()

    def mean(self):
        r"""Estimated mean :math:`\mathrm X\boldsymbol\beta`."""
        return self.m

    @property
    def scale(self):
        if self._fix['scale']:
            return self._scale
        return LMMCore.scale.fget(self)

    @scale.setter
    def scale(self, v):
        self._scale = v

    def unfix(self, var_name):
        r"""Enable the optimisation of a given variable.

        Parameters
        ----------
        var_name : str
            Possible values are `delta`, and TODO.
        """
        if var_name == 'delta':
            super(LMM, self).unfix('logistic')
        else:
            self._fix[var_name] = False

    def value(self):
        self.delta = self._get_delta()
        return self.lml()

    @property
    def X(self):
        return super(LMM, self).X.fget()
