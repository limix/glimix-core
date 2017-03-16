from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import sign
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon
from optimix import Function, Scalar

from liknorm import LikNormMachine

from ..ep import EP


class GLMM(EP, Function):
    r"""Expectation Propagation for Generalised Gaussian Processes.

    Args:
        y (array_like): outcome variable.
        lik_name (str): likelihood name.
        mean (:class:`optimix.Function`): mean function.
                                          (Refer to :doc:`mean`.)
        cov (:class:`optimix.Function`): covariance function.
                                         (Refer to :doc:`cov`.)

    Example
    -------
    """

    # .. doctest::
    #
    #     >>> from numpy.random import RandomState
    #     >>>
    #     >>> from limix_inference.example import offset_mean
    #     >>> from limix_inference.example import linear_eye_cov
    #     >>> from limix_inference.ggp import GLMM
    #     >>> from limix_inference.lik import BernoulliProdLik
    #     >>> from limix_inference.link import LogitLink
    #     >>> from limix_inference.random import GGPSampler
    #     >>>
    #     >>> random = RandomState(1)
    #     >>>
    #     >>> lik = BernoulliProdLik(LogitLink())
    #     >>> mean = offset_mean()
    #     >>> cov = linear_eye_cov()
    #     >>>
    #     >>> y = GGPSampler(lik, mean, cov).sample(random)
    #     >>>
    #     >>> ggp = GLMM(y, 'bernoulli', mean, cov)
    #     >>> print('Before: %.4f' % ggp.feed().value())
    #     Before: -65.8090
    #     >>> ggp.feed().maximize(progress=False)
    #     >>> print('After: %.2f' % ggp.feed().value())
    #     After: -65.39


    def __init__(self, y, lik_name, X, QS):
        super(GLMM, self).__init__()
        Function.__init__(self, logitscale=Scalar(0.0), logitdelta=Scalar(0.0))

        self._logger = logging.getLogger(__name__)

        self._y = y
        self._X = X
        self._QS = QS

        self._machine = LikNormMachine(lik_name, 500)
        self.set_nodata()

    def __del__(self):
        if hasattr(self, '_machine'):
            self._machine.finish()

    def _compute_moments(self):
        tau = self._cav.tau
        eta = self._cav.eta
        self._machine.moments(self._y, eta, tau, self._moments)

    def value(self):
        r"""Log of the marginal likelihood.

        Args:
            mean (array_like): realised mean.
            cov (array_like): realised covariance.

        Returns:
            float: log of the marginal likelihood.
        """
        try:
            self._initialize(mean, cov)
            self._params_update()
            lml = self._lml()
        except (ValueError, LinAlgError) as e:
            print(e)
            print("value: returning large value.\n")
            lml = -1 / epsilon.small
        return lml

    def gradient(self, mean, cov, gmean, gcov):  # pylint: disable=W0221
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
            self._initialize(mean, cov)
            self._params_update()

            grad = [self._lml_derivative_over_cov(gc) for gc in gcov]
            grad += [self._lml_derivative_over_mean(gm) for gm in gmean]

            return grad
        except (ValueError, LinAlgError) as e:
            print(e)
            print("gradient: returning large value.\n")
            v = self.variables().select(fixed=False)
            return [-sign(v.get(i).value) / epsilon.small for i in v]
