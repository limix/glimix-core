from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import (asarray, clip, concatenate, dot, exp, inf, log, ndarray,
                   zeros)
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon

from liknorm import LikNormMachine
from optimix import Function, Scalar, Vector
from optimix.optimize import BadSolutionError

from ..ep import EP


class GLMM(EP, Function):
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

    Args:
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
        ...     for j in range(ntrials[i]):
        ...         successes[i] += int(z[i] + 0.5 * random.randn() > 0)
        >>>
        >>> y = (successes, ntrials)
        >>>
        >>> QS = economic_qs(K)
        >>> glmm = GLMM(y, 'binomial', X, QS)
        >>> print('Before: %.4f' % glmm.feed().value())
        Before: -151.5476
        >>> glmm.feed().maximize(progress=False)
        >>> print('After: %.2f' % glmm.feed().value())
        After: -147.53
    """

    def __init__(self, y, lik_name, X, QS):
        super(GLMM, self).__init__()
        Function.__init__(
            self,
            beta=Vector(zeros(X.shape[1])),
            logscale=Scalar(0.0),
            logitdelta=Scalar(0.0))

        self._logger = logging.getLogger(__name__)

        logscale = self.variables()['logscale']
        logscale.bounds = (-5, 4.0)
        logscale.listen(self._clear_cache)

        logitdelta = self.variables()['logitdelta']
        logitdelta.bounds = (-30.0, +30.0)
        logitdelta.listen(self._clear_cache)

        self.variables()['beta'].listen(self._clear_cache)

        if isinstance(y, list):
            y = tuple(y)
        elif not isinstance(y, tuple):
            y = (y, )

        self._y = tuple([asarray(i, float) for i in y])
        self._X = X

        if not isinstance(QS, tuple):
            raise ValueError("QS must be a tuple.")

        if not isinstance(QS[0], tuple):
            raise ValueError("QS[0] must be a tuple.")

        Q = concatenate(QS[0], axis=1)
        S = zeros(Q.shape[0], dtype=float)
        S[:QS[1].shape[0]] = QS[1]

        self._QS = (Q, S)

        self._machine = LikNormMachine(lik_name, 500)
        self.set_nodata()

    def _clear_cache(self, _):
        self._need_params_update = True

    def __del__(self):
        if hasattr(self, '_machine'):
            self._machine.finish()

    def _compute_moments(self):
        tau = self._cav['tau']
        eta = self._cav['eta']
        self._machine.moments(self._y, eta, tau, self._moments)

    def mean(self):
        return dot(self._X, self.beta)

    def fix(self, var_name):
        if var_name == 'scale':
            Function.fix(self, 'logscale')
        elif var_name == 'delta':
            Function.fix(self, 'logitdelta')
        elif var_name == 'beta':
            Function.fix(self, 'beta')
        else:
            raise ValueError("Unknown parameter name %s." % var_name)
        self._clear_cache(None)

    def unfix(self, var_name):
        if var_name == 'scale':
            Function.unfix(self, 'logscale')
        elif var_name == 'delta':
            Function.unfix(self, 'logitdelta')
        elif var_name == 'beta':
            Function.unfix(self, 'beta')
        else:
            raise ValueError("Unknown parameter name %s." % var_name)
        self._clear_cache(None)

    @property
    def scale(self):
        return float(exp(self.variables().get('logscale').value))

    @scale.setter
    def scale(self, v):
        self.variables().get('logscale').value = log(v)
        self._clear_cache(None)

    @property
    def delta(self):
        return float(1 / (1 + exp(-self.variables().get('logitdelta').value)))

    @delta.setter
    def delta(self, v):
        v = clip(v, epsilon.small, 1 - epsilon.small)
        self.variables().get('logitdelta').value = log(v / (1 - v))
        self._clear_cache(None)

    @property
    def beta(self):
        return self.variables().get('beta').value

    @beta.setter
    def beta(self, v):
        self.variables().get('beta').value = v
        self._clear_cache(None)

    def _eigval_derivative_over_logscale(self):
        x = self.variables().get('logscale').value
        d = self.delta
        E = self._QS[1]
        return exp(x) * ((1 - d) * E + d)

    def _eigval_derivative_over_logitdelta(self):
        x = self.variables().get('logitdelta').value
        s = self.scale
        c = (s * exp(x) / (1 + exp(-x))**2)
        E = self._QS[1]
        cond = abs(E.max())/abs(E.min())
        if cond > 1e5:
            self._logger.info("Conditioning number too high: %.5f", cond)
        res = c * (1 - E)
        return res

    def _S(self):
        s = self.scale
        d = self.delta
        return s * ((1 - d) * self._QS[1] + d)

    def value(self):
        r"""Log of the marginal likelihood.

        Args:
            mean (array_like): realised mean.
            cov (array_like): realised covariance.

        Returns:
            float: log of the marginal likelihood.
        """
        try:
            self._initialize(self.mean(), (self._QS[0], self._S()))
            self._params_update()
            lml = self._lml()
        except (ValueError, LinAlgError) as e:
            self._logger.info(str(e))
            self._logger.info("Beta: %s", str(self.beta))
            self._logger.info("Delta: %.10f", self.delta)
            self._logger.info("Scale: %.10f", self.scale)
            raise BadSolutionError(str(e))
        return lml

    def gradient(self):  # pylint: disable=W0221
        r"""Gradient of the log of the marginal likelihood.

        Args:
            mean (array_like): realised mean.
            cov (array_like): realised cov.
            gmean (array_like): realised mean derivative.
            gcov (array_like): realised covariance derivative.

        Returns:
            list: derivatives.
        """
        try:
            self._initialize(self.mean(), (self._QS[0], self._S()))
            self._params_update()

            dS0 = self._eigval_derivative_over_logitdelta()
            dS1 = self._eigval_derivative_over_logscale()

            grad = [
                self._lml_derivative_over_cov((self._QS[0], dS0)),
                self._lml_derivative_over_cov((self._QS[0], dS1)),
                self._lml_derivative_over_mean(self._X)
            ]
        except (ValueError, LinAlgError) as e:
            self._logger.info(str(e))
            self._logger.info("Beta: %s", str(self.beta))
            self._logger.info("Delta: %.10f", self.delta)
            self._logger.info("Scale: %.10f", self.scale)
            raise BadSolutionError(str(e))

        return dict(logitdelta=grad[0], logscale=grad[1], beta=grad[2])
