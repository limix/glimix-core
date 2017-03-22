from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import asarray, dot, exp, sign, zeros, log, ndarray
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon

from liknorm import LikNormMachine
from optimix import Function, Scalar, Vector

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

        >>> from numpy.random import RandomState
        >>>
        >>> from limix_inference.example import offset_mean
        >>> from limix_inference.example import linear_eye_cov
        >>> from limix_inference.ggp import ExpFamGP
        >>> from limix_inference.lik import BernoulliProdLik
        >>> from limix_inference.link import LogitLink
        >>> from limix_inference.random import GGPSampler
        >>>
        >>> random = RandomState(1)
        >>>
        >>> lik = BernoulliProdLik(LogitLink())
        >>> mean = offset_mean()
        >>> cov = linear_eye_cov()
        >>>
        >>> y = GGPSampler(lik, mean, cov).sample(random)
        >>>
        >>> ggp = ExpFamGP(y, 'bernoulli', mean, cov)
        >>> print('Before: %.4f' % ggp.feed().value())
        Before: -65.8090
        >>> ggp.feed().maximize(progress=False)
        >>> print('After: %.2f' % ggp.feed().value())
        After: -65.39
    """

    def __init__(self, y, lik_name, X, QS):
        super(GLMM, self).__init__()
        Function.__init__(
            self,
            beta=Vector(zeros(X.shape[1])),
            logscale=Scalar(0.0),
            logitdelta=Scalar(0.0))

        self._logger = logging.getLogger(__name__)

        if not isinstance(y, tuple):
            y = (y, )

        self._y = tuple([asarray(i, float) for i in y])
        self._X = X

        if not isinstance(QS, tuple):
            raise ValueError("QS must be a tuple.")

        if not isinstance(QS[0], ndarray):
            raise ValueError("QS[0] has to be numpy.ndarray.")

        if not isinstance(QS[1], ndarray):
            raise ValueError("QS[1] has to be numpy.ndarray.")

        self._QS = QS

        self._machine = LikNormMachine(lik_name, 500)
        self.set_nodata()

    def __del__(self):
        if hasattr(self, '_machine'):
            self._machine.finish()

    def _compute_moments(self):
        tau = self._cav['tau']
        eta = self._cav['eta']
        self._machine.moments(self._y, eta, tau, self._moments)

    def mean(self):
        return dot(self._X, self.beta)

    @property
    def scale(self):
        return exp(self.variables().get('logscale').value)

    @scale.setter
    def scale(self, v):
        self.variables().get('logscale').value = log(v)

    @property
    def delta(self):
        return 1 / (1 + exp(-self.variables().get('logitdelta').value))

    @property
    def beta(self):
        return self.variables().get('beta').value

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
        return c * (1 - E)

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
            lml = -1 / epsilon.small
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

            grad = dict()
            grad['logitdelta'] = self._lml_derivative_over_cov((self._QS[0], dS0))
            grad['logscale'] = self._lml_derivative_over_cov((self._QS[0], dS1))
            grad['beta'] = self._lml_derivative_over_mean(self._X)

            return grad
        except (ValueError, LinAlgError) as e:
            self._logger.info(str(e))
            v = self.variables().select(fixed=False)
            return [-sign(v.get(i).value) / epsilon.small for i in v]
