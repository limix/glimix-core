from __future__ import absolute_import, division, unicode_literals

import logging
from math import fsum

from numpy.linalg import norm

from numpy import (inf, isfinite, empty, maximum)

from numpy_sugar import epsilon

from .cavity import Cavity
from .posterior import Posterior
from .site import Site

MAX_ITERS = 30
RTOL = epsilon.small
ATOL = epsilon.small


class EP(object):
    r"""Generic EP implementation.

    Let :math:`\mathrm Q \mathrm S \mathrm Q^{\intercal}` be the economic
    eigendecomposition of the genetic covariance matrix.
    Let :math:`\mathrm U\mathrm S\mathrm V^{\intercal}` be the singular value
    decomposition of the user-provided covariates :math:`\mathrm M`. We define

    .. math::

        \mathrm K = v ((1-\delta)\mathrm Q \mathrm S \mathrm Q^{\intercal} +
                    \delta \mathrm I)

    as the covariance of the prior distribution. As such,
    :math:`v` and :math:`\delta` refer to :py:attr:`_v` and :py:attr:`_delta`
    class attributes, respectively. We also use the following variables for
    convenience:

    .. math::
        :nowrap:

        \begin{eqnarray}
            \sigma_b^2          & = & v (1-\delta) \\
            \sigma_{\epsilon}^2 & = & v \delta
        \end{eqnarray}

    The covariate effect-sizes is given by :math:`\boldsymbol\beta`, which
    implies

    .. math::

        \mathbf m = \mathrm M \boldsymbol\beta

    The prior is thus defined as

    .. math::

        \mathcal N(\mathbf z ~|~ \mathbf m; \mathrm K)

    and the marginal likelihood is given by

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | g(\mathrm E[y_i | z_i])=z_i)
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    However, the singular value decomposition of the covariates allows us to
    automatically remove dependence between covariates, which would create
    infinitly number of :math:`\boldsymbol\beta` that lead to global optima.
    Let us define

    .. math::

        \tilde{\boldsymbol\beta} = \mathrm S^{1/2} \mathrm V^{\intercal}
                                    \boldsymbol\beta

    as the covariate effect-sizes we will effectively work with during the
    optimization process. Let us also define the

    .. math::

        \tilde{\mathrm M} = \mathrm U \mathrm S^{1/2}

    as the redundance-free covariates. Naturally,

    .. math::

        \mathbf m = \tilde{\mathrm M} \tilde{\boldsymbol\beta}

    In summary, we will optimize :math:`\tilde{\boldsymbol{\beta}}`, even
    though the user will be able to retrieve the corresponding
    :math:`\boldsymbol{\beta}`.


    Let

    .. math::

        \mathrm{KL}[p(y_i|z_i) p_{-}(z_i|y_i)_{\text{EP}} ~|~
            p(y_i|z_i)_{\text{EP}} p_{-}(z_i|y_i)_{\text{EP}}]

    be the KL divergence we want to minimize at each EP iteration.
    The left-hand side can be described as
    :math:`\hat c_i \mathcal N(z_i | \hat \mu_i; \hat \sigma_i^2)`


    Args:
        M (array_like): :math:`\mathrm M` covariates.
        Q (array_like): :math:`\mathrm Q` of the economic
                        eigendecomposition.
        S (array_like): :math:`\mathrm S` of the economic
                        eigendecomposition.
        overdispersion (bool): `True` for :math:`\sigma_{\epsilon}^2 \ge 0`,
                `False` for :math:`\sigma_{\epsilon}^2=0`.
        QSQt (array_like): :math:`\mathrm Q \mathrm S
                        \mathrm Q^{\intercal}` in case this has already
                        been computed. Defaults to `None`.


    Attributes:
        _v (float): Total variance :math:`v` from the prior distribution.
        _delta (float): Fraction of the total variance due to the identity
                        matrix :math:`\mathrm I`.
        _loghz (array_like): This is :math:`\log(\hat c)` for each site.
        _hmu (array_like): This is :math:`\hat \mu` for each site.
        _hvar (array_like): This is :math:`\hat \sigma^2` for each site.

    """
    def __init__(self):
        self._logger = logging.getLogger(__name__)

        self._site = None
        self._psite = None

        self._cav = None
        self._posterior = None

        self._moments = {'log_zeroth': None,
                         'mean': None,
                         'variance': None}

        self._initialized = False

    def set_prior_mean(self, mean):
        self._posterior.set_prior_mean(mean)

    def set_prior_cov(self, cov):
        self._posterior.set_prior_cov(cov)

    def _compute_moments(self):
        raise NotImplementedError

    def _initialize(self, mean, cov):
        self._logger.debug("EP parameters initialization.")

        nsamples = len(mean)

        self._site = Site(nsamples)
        self._psite = Site(nsamples)

        self._cav = Cavity(nsamples)
        self._posterior = Posterior(self._site)
        self._posterior.set_prior_mean(mean)
        self._posterior.set_prior_cov(cov)

        self._moments = {'log_zeroth': empty(nsamples),
                         'mean': empty(nsamples),
                         'variance': empty(nsamples)}

        # if self._initialized:
        #     self._posterior.update(mean, cov)
        # else:
        self._posterior.initialize()

    def _lml_components(self):
        return [0, 1]

    def _lml(self):
        v = fsum(self._lml_components())
        if not isfinite(v):
            raise ValueError("LML should not be %f." % v)
        return v

    def _params_update(self):
        self._logger.debug('EP parameters update loop has started.')

        i = 0
        tol = inf
        while i < MAX_ITERS and norm(self._site.tau - self._psite.tau) < tol:
            self._psite.tau[:] = self._site.tau
            self._psite.eta[:] = self._site.eta

            self._cav.tau[:] = maximum(self._posterior.tau - self._site.tau, 0)
            self._cav.eta[:] = self._posterior.eta - self._site.eta
            self._compute_moments()

            self._site.update(self._moments['mean'],
                              self._moments['variance'],
                              self._cav.eta,
                              self._cav.tau)

            self._posterior.update()

            tol = RTOL * norm(self._psite.tau) + ATOL
            i += 1

        if i == MAX_ITERS:
            self._logger.warning('Maximum number of EP iterations has' +
                                 ' been attained.')

        self._logger.debug('EP loop has performed %d iterations.', i)
