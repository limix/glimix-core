from __future__ import absolute_import, division, unicode_literals

from copy import copy

from liknorm import LikNormMachine
from numpy import asarray, clip, dot, exp, inf, log, zeros
from numpy.linalg import LinAlgError

from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, sum2diag
from optimix import Function, Scalar, Vector

from ..check import check_covariates, check_economic_qs, check_outcome
from ..ep import EPLinearKernel
from ..io import eprint, wprint
from .expfam import GLMMExpFam


class GLMM(object):
    r"""Expectation Propagation for Generalised Gaussian Processes.

    Let

    .. math::

        \mathrm Q \mathrm E \mathrm Q^{\intercal}
        = \mathrm G\mathrm G^{\intercal}

    be the eigen decomposition of the random effect's covariance.
    It turns out that the prior covariance of GLMM can be described as

    .. math::

        \mathrm Q s((1-\delta)\mathrm E
        + \delta\mathrm I) \mathrm Q^{\intercal}.

    This means that :math:`\mathrm Q` does not change during inference, and
    this fact is used here to speed-up EP inference for GLMM.

    Parameters
    ----------
        y (array_like): outcome variable.
        lik_name (str): likelihood name.
        mean (:class:`optimix.Function`): mean function.
                                          (Refer to :doc:`mean`.)
        cov (:class:`optimix.Function`): covariance function.
                                         (Refer to :doc:`cov`.)

    Example
    -------

    .. doctest::

        >>> from numpy import dot, sqrt, zeros
        >>> from numpy.random import RandomState
        >>>
        >>> from numpy_sugar.linalg import economic_qs
        >>>
        >>> from glimix_core.glmm import GLMM
        >>>
        >>> random = RandomState(0)
        >>> nsamples = 50
        >>>
        >>> X = random.randn(50, 2)
        >>> G = random.randn(50, 100)
        >>> K = dot(G, G.T)
        >>> ntrials = random.randint(1, 100, nsamples)
        >>> z = dot(G, random.randn(100)) / sqrt(100)
        >>>
        >>> successes = zeros(len(ntrials), int)
        >>> for i in range(len(ntrials)):
        ...     successes[i] = sum(z[i] + 0.2 * random.randn(ntrials[i]) > 0)
        >>>
        >>> y = (successes, ntrials)
        >>>
        >>> QS = economic_qs(K)
        >>> glmm = GLMM(y, 'binomial', X, QS)
        >>> print('Before: %.2f' % glmm.lml())
        Before: -95.19
        >>> glmm.fit(verbose=False)
        >>> print('After: %.2f' % glmm.lml())
        After: -92.24
    """

    def __init__(self, y, lik_name, X, QS):

        X = asarray(X, float)
        y = _normalise_outcome(y)
        y = check_outcome(y, lik_name)
        X = check_covariates(X)
        QS = check_economic_qs(QS)

        self._func = GLMMExpFam(y, lik_name, X, QS)

        self._func.set_variable_bounds('logscale', (log(1e-3), 7.))
        self._func.set_variable_bounds('logitdelta', (-inf, +inf))

        self._QS = QS

    def __copy__(self):
        cls = self.__class__
        glmm = cls.__new__(cls)
        glmm.__dict__['_func'] = copy(self.function)
        glmm.__dict__['_QS'] = self._QS
        return glmm

    @property
    def beta(self):
        return self._func.beta

    @beta.setter
    def beta(self, v):
        self._func.beta = v

    def copy(self):
        return copy(self)

    def covariance(self):
        Q0 = self._QS[0][0]
        S0 = self._QS[1]
        return sum2diag(dot(ddot(Q0, self.v0 * S0), Q0.T), self.v1)

    @property
    def delta(self):
        return 1 / (1 + exp(-self._func.logitdelta))

    @delta.setter
    def delta(self, v):
        v = clip(v, epsilon.large, 1 - epsilon.large)
        self._func.logitdelta = log(v / (1 - v))

    def fix(self, var_name):
        self._func.fix(_to_internal_name(var_name))

    def fit(self, verbose=True):
        self._func.feed().maximize(verbose=verbose)

    @property
    def function(self):
        return self._func

    def lml(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        return self._func.value()

    def mean(self):
        return self._func.mean()

    @property
    def scale(self):
        return exp(self._func.logscale)

    @scale.setter
    def scale(self, v):
        self._func.logscale = log(v)

    def unfix(self, var_name):
        self._func.unfix(_to_internal_name(var_name))

    @property
    def v0(self):
        return self.scale * (1 - self.delta)

    @property
    def v1(self):
        return self.scale * self.delta


def _normalise_outcome(y):
    if isinstance(y, list):
        y = tuple(y)
    elif not isinstance(y, tuple):
        y = (y, )
    return tuple([asarray(i, float) for i in y])


def _to_internal_name(name):
    translation = dict(scale='logscale', delta='logitdelta', beta='beta')
    return translation[name]
