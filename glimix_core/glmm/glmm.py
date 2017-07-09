from __future__ import absolute_import, division, unicode_literals

import logging

from liknorm import LikNormMachine
from numpy import (asarray, clip, concatenate, dot, exp, inf, log, ndarray,
                   zeros)
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon
from optimix import Function, Scalar, Vector

from ..ep import EPLinearKernel


class GLMM(EPLinearKernel, Function):
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
        ...     successes[i] = sum(z[i] + 0.2 * random.randn(ntrials[i]) > 0)
        >>>
        >>> y = (successes, ntrials)
        >>>
        >>> QS = economic_qs(K)
        >>> glmm = GLMM(y, 'binomial', X, QS)
        >>> print('Before: %.4f' % glmm.feed().value())
        Before: -95.1854
        >>> glmm.feed().maximize(verbose=False)
        >>> print('After: %.2f' % glmm.feed().value())
        After: -92.24
    """

    def __init__(self, y, lik_name, X, QS):
        super(GLMM, self).__init__(X.shape[0])
        Function.__init__(
            self,
            beta=Vector(zeros(X.shape[1])),
            logscale=Scalar(0.0),
            logitdelta=Scalar(0.0))

        self._logger = logging.getLogger(__name__)

        logscale = self.variables()['logscale']
        logscale.bounds = (log(1e-3), 7.)
        logscale.listen(self._set_need_prior_update)

        logitdelta = self.variables()['logitdelta']
        logitdelta.bounds = (-inf, +inf)
        logitdelta.listen(self._set_need_prior_update)

        self.variables()['beta'].listen(self._set_need_prior_update)

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

        self._QS = QS

        self._machine = LikNormMachine(lik_name, 500)
        self._need_prior_update = True
        self.set_nodata()

    def _set_need_prior_update(self, _=None):
        self._need_prior_update = True

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
        self._set_need_prior_update()

    def unfix(self, var_name):
        if var_name == 'scale':
            Function.unfix(self, 'logscale')
        elif var_name == 'delta':
            Function.unfix(self, 'logitdelta')
        elif var_name == 'beta':
            Function.unfix(self, 'beta')
        else:
            raise ValueError("Unknown parameter name %s." % var_name)
        self._set_need_prior_update()

    @property
    def scale(self):
        return float(exp(self.variables().get('logscale').value))

    @scale.setter
    def scale(self, v):
        self.variables().get('logscale').value = log(v)
        self._set_need_prior_update()

    @property
    def delta(self):
        return float(1 / (1 + exp(-self.variables().get('logitdelta').value)))

    @delta.setter
    def delta(self, v):
        v = clip(v, epsilon.large, 1 - epsilon.large)
        self.variables().get('logitdelta').value = log(v / (1 - v))
        self._set_need_prior_update()

    @property
    def beta(self):
        return asarray(self.variables().get('beta').value, float)

    @beta.setter
    def beta(self, v):
        self.variables().get('beta').value = v
        self._set_need_prior_update()

    def value(self):
        r"""Log of the marginal likelihood.

        Args:
            mean (array_like): realised mean.
            cov (array_like): realised covariance.

        Returns:
            float: log of the marginal likelihood.
        """
        try:
            if self._need_prior_update:
                cov = dict(QS=self._QS, scale=self.scale, delta=self.delta)
                self._set_prior(self.mean(), cov)
                self._need_prior_update = False

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
            if self._need_prior_update:
                cov = dict(QS=self._QS, scale=self.scale, delta=self.delta)
                self._set_prior(self.mean(), cov)
                self._need_prior_update = False

            v = self.variables().get('logitdelta').value
            x = self.variables().get('logscale').value
            g = self._lml_derivatives(self._X)
            ev = exp(-v)
            grad = [
                g['delta'] * (ev / (1 + ev)) / (1 + ev), g['scale'] * exp(x),
                g['mean']
            ]
        except (ValueError, LinAlgError) as e:
            self._logger.info(str(e))
            self._logger.info("Beta: %s", str(self.beta))
            self._logger.info("Delta: %.10f", self.delta)
            self._logger.info("Scale: %.10f", self.scale)
            raise BadSolutionError(str(e))

        return dict(logitdelta=grad[0], logscale=grad[1], beta=grad[2])


class BadSolutionError(Exception):
    pass
