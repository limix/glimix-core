from __future__ import absolute_import, division, unicode_literals

from liknorm import LikNormMachine
from numpy import sign
from numpy.linalg import LinAlgError

from numpy_sugar import epsilon
from numpy_sugar.linalg import economic_qs
from optimix import FunctionReduce

from ..ep import EP
from ..util import wprint


class ExpFamGP(FunctionReduce):
    r"""Expectation Propagation for Generalised Gaussian Processes.

    Parameters
    ----------
    y : array_like
        Outcome variable.
    lik_name : str
        Likelihood name.
    mean : :class:`optimix.Function`
        Mean function. (Refer to :doc:`mean`.)
    cov : :class:`optimix.Function`
        Covariance function. (Refer to :doc:`cov`.)

    Example
    -------

    .. doctest::

        >>> from numpy.random import RandomState
        >>>
        >>> from glimix_core.example import offset_mean
        >>> from glimix_core.example import linear_eye_cov
        >>> from glimix_core.ggp import ExpFamGP
        >>> from glimix_core.lik import BernoulliProdLik
        >>> from glimix_core.link import LogitLink
        >>> from glimix_core.random import GGPSampler
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
        Before: -6.9802
        >>> ggp.feed().maximize(verbose=False)
        >>> print('After: %.2f' % ggp.feed().value())
        After: -6.55
    """

    def __init__(self, y, lik_name, mean, cov):
        if isinstance(y, tuple):
            n = len(y[0])
        else:
            n = len(y)
        FunctionReduce.__init__(self, [mean, cov], name='ExpFamGP')

        self._y = y
        self._mean = mean
        self._cov = cov
        self._ep = EP(n)
        self._ep.set_compute_moments(self.compute_moments)

        self._mean_value = None
        self._cov_value = None

        self._machine = LikNormMachine(lik_name, 500)

    def __del__(self):
        if hasattr(self, '_machine'):
            self._machine.finish()

    def compute_moments(self, eta, tau, moments):
        self._machine.moments(self._y, eta, tau, moments)

    def value_reduce(self, values):
        r"""Log of the marginal likelihood.

        Parameters
        ----------
        mean : array_like
            Realised mean.
        cov : array_like
            Realised covariance.

        Returns
        -------
        float
            Log of the marginal likelihood.
        """
        mean = values['ExpFamGP[0]']
        cov = values['ExpFamGP[1]']
        try:
            self._ep.set_prior(mean, dict(QS=economic_qs(cov)))
            lml = self._ep.lml()
        except (ValueError, LinAlgError) as e:
            wprint(str(e))
            lml = -1 / epsilon.small
        return lml

    def gradient_reduce(self, values, gradients):
        r"""Gradient of the log of the marginal likelihood.

        Parameters
        ----------
        mean : array_like
            Realised mean.
        cov : array_like
            Realised cov.
        gmean : array_like
            Realised mean derivative.
        gcov : array_like
            Realised covariance derivative.

        Returns
        -------
        list
            Derivatives.
        """

        mean = values['ExpFamGP[0]']
        cov = values['ExpFamGP[1]']
        gmean = gradients['ExpFamGP[0]']
        gcov = gradients['ExpFamGP[1]']

        try:
            self._ep.set_prior(mean, dict(QS=economic_qs(cov)))

            grad = dict()

            for n, g in iter(gmean.items()):
                grad['ExpFamGP[0].' + n] = self._ep.lml_derivative_over_mean(g)

            for n, g in iter(gcov.items()):
                QS = economic_qs(g)
                grad['ExpFamGP[1].' + n] = self._ep.lml_derivative_over_cov(QS)

            return grad
        except (ValueError, LinAlgError) as e:
            wprint(str(e))
            v = self.variables().select(fixed=False)
            return {i: -sign(v.get(i).value) / epsilon.small for i in v}
