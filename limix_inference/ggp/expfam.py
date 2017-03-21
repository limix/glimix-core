from __future__ import absolute_import, division, unicode_literals

import logging

from liknorm import LikNormMachine
from numpy import sign
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs

from optimix import FunctionReduce

from ..ep import EP


class ExpFamGP(EP, FunctionReduce):
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

    def __init__(self, y, lik_name, mean, cov):
        super(ExpFamGP, self).__init__()
        FunctionReduce.__init__(self, [mean, cov], name='ExpFamGP')

        self._logger = logging.getLogger(__name__)

        self._y = y
        self._mean = mean
        self._cov = cov

        self._mean_value = None
        self._cov_value = None

        self._machine = LikNormMachine(lik_name, 500)

    def __del__(self):
        if hasattr(self, '_machine'):
            self._machine.finish()

    def _compute_moments(self):
        tau = self._cav['tau']
        eta = self._cav['eta']
        self._machine.moments(self._y, eta, tau, self._moments)

    # def value(self, mean, cov):
    def value_reduce(self, values):  # pylint: disable=R0201
        r"""Log of the marginal likelihood.

        Args:
            mean (array_like): realised mean.
            cov (array_like): realised covariance.

        Returns:
            float: log of the marginal likelihood.
        """
        mean = values['ExpFamGP[0]']
        cov = values['ExpFamGP[1]']
        QS = economic_qs(cov)
        try:
            self._initialize(mean, (QS[0][0], QS[1]))
            self._params_update()
            lml = self._lml()
        except (ValueError, LinAlgError) as e:
            self._logger.info(str(e))
            lml = -1 / epsilon.small
        return lml

    # def gradient(self, mean, cov, gmean, gcov):  # pylint: disable=W0221
    def gradient_reduce(self, values, gradients):
        r"""Gradient of the log of the marginal likelihood.

        Args:
            mean (array_like): realised mean.
            cov (array_like): realised cov.
            gmean (array_like): realised mean derivative.
            gcov (array_like): realised covariance derivative.

        Returns:
            list: derivatives.
        """

        mean = values['ExpFamGP[0]']
        cov = values['ExpFamGP[1]']
        gmean = gradients['ExpFamGP[0]']
        gcov = gradients['ExpFamGP[1]']

        try:
            QS = economic_qs(cov)
            self._initialize(mean, (QS[0][0], QS[1]))
            self._params_update()

            grad = dict()

            for n, g in iter(gmean.items()):
                grad['ExpFamGP[0].' + n] = self._lml_derivative_over_mean(g)

            for n, g in iter(gcov.items()):
                QS = economic_qs(g)
                grad['ExpFamGP[1].' + n] = self._lml_derivative_over_cov(
                    (QS[0][0], QS[1]))

            return grad
        except (ValueError, LinAlgError) as e:
            self._logger.info(str(e))
            v = self.variables().select(fixed=False)
            return {i: -sign(v.get(i).value) / epsilon.small for i in v}
