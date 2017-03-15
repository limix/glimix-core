from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import sign
from numpy.linalg import LinAlgError
from numpy_sugar import epsilon
from optimix import Composite

from liknorm import LikNormMachine

from ..ep import EP


class ExpFamGP(EP, Composite):
    r"""Expectation Propagation for exponential family distributions.

    Models

    .. math::

        y_i ~|~ z_i \sim \text{ExpFam}(y_i ~|~ \mu_i = g(z_i))

    for

    .. math::

        \mathbf z \sim \mathcal N\big(~~ \mathrm M^\intercal \boldsymbol\beta;~
            \sigma_b^2 \mathrm Q_0 \mathrm S_0 \mathrm Q_0^{\intercal} +
                    \sigma_{\epsilon}^2 \mathrm I ~~\big)

    where :math:`\mathrm Q_0 \mathrm S_0 \mathrm Q_0^\intercal`
    is the economic eigen decomposition of a semi-definite positive matrix,
    :math:`g(\cdot)` is a link function, and :math:`\text{ExpFam}(\cdot)` is
    an exponential-family distribution.

    For convenience, let us define the following variables:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \sigma_b^2          & = & v (1-\delta) \\
            \sigma_{\epsilon}^2 & = & v \delta\\
            \mathbf m           & = & \mathrm M \boldsymbol\beta \\
            \mathrm K           & = & \sigma_b^2 \mathrm Q_0 \mathrm S_0
                                      \mathrm Q_0^{\intercal} +
                                      \sigma_{\epsilon}^2 \mathrm I
        \end{eqnarray}

    Args:
        prodlik (object): likelihood product.
        covariates (array_like): fixed-effect covariates :math:`\mathrm M`.
        Q0 (array_like): eigenvectors of positive eigenvalues.
        Q1 (array_like): eigenvectors of zero eigenvalues.
        S0 (array_like): positive eigenvalues.

    Example
    ^^^^^^^

    .. doctest::

        >>> from numpy.random import RandomState
        >>>
        >>> from limix_inference.example import offset_mean
        >>> from limix_inference.example import linear_eye_cov
        >>> from limix_inference.ggp import ExpFamGP
        >>> from limix_inference.lik import BernoulliProdLik
        >>> from limix_inference.link import LogitLink
        >>> from limix_inference.random import GLMMSampler
        >>>
        >>> random = RandomState(0)
        >>>
        >>> lik = BernoulliProdLik(LogitLink())
        >>> mean = offset_mean()
        >>> cov = linear_eye_cov()
        >>>
        >>> y = GLMMSampler(lik, mean, cov).sample(random)
        >>>
        >>> ggp = ExpFamGP(y, 'bernoulli', mean, cov)
        >>> ggp.feed().maximize()
        >>> print('%.2f' % ggp.feed().value())
        -69.06
    """

    def __init__(self, y, lik_name, mean, cov):
        super(ExpFamGP, self).__init__()
        Composite.__init__(self, mean=mean, cov=cov)

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
        tau = self._cav.tau
        eta = self._cav.eta
        self._machine.moments(self._y, eta, tau, self._moments)

    def value(self, mean, cov):
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

    # def copy(self):
    #     # pylint: disable=W0212
    #     ep = ExpFamGP.__new__(ExpFamGP)
    #     self._copy_to(ep)
    #
    #     ep._machine = LikNormMachine(self._lik_prod.name, 500)
    #     ep._lik_prod = self._lik_prod
    #
    #     ep._phenotype = self._phenotype
    #     ep._tbeta = self._tbeta.copy()
    #     ep.delta = self.delta
    #     self.v = self.v
    #
    #     return ep
