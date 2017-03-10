from __future__ import absolute_import, division, unicode_literals

import logging

from numpy_sugar import is_all_finite
from numpy import ascontiguousarray, clip, full
from numpy.linalg import lstsq

from ...liknorm import LikNormMachine
from ...lmm import FastLMM
from .ep import EP


class ExpFamEP(EP):
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
    """

    def __init__(self,
                 prodlik,
                 covariates,
                 QS,
                 overdispersion=True,
                 options=None):
        covariates = ascontiguousarray(covariates, float)

        if options is None:
            options = dict(rank_norm=False)
        self._options = options

        super(ExpFamEP, self).__init__(covariates, QS[0][0], QS[1],
                                       overdispersion)
        self._logger = logging.getLogger(__name__)

        self._Q1 = QS[0][1]
        self._machine = LikNormMachine(500)
        self._prodlik = prodlik

        h2, m = _initialize(prodlik, covariates, QS)

        n = prodlik.sample_size

        self._phenotype = prodlik
        self._tbeta = lstsq(self._tM, full(n, m))[0]

        if overdispersion:
            self.delta = 1 - h2
            self.v = 1.
        else:
            self.delta = 0
            self.v = (h2 * prodlik.latent_variance) / (1 - h2)

    @property
    def options(self):
        return self._options

    def _tilted_params(self):
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz
        self._machine.moments(self._prodlik.name, self._phenotype.ytuple, ceta,
                              ctau, lmom0, self._hmu, self._hvar)
        if not is_all_finite(lmom0):
            raise ValueError("lmom0 should not be %s." % str(lmom0))
        if not is_all_finite(self._hmu):
            raise ValueError("hmu should not be %s." % str(self._hmu))
        if not is_all_finite(self._hvar):
            raise ValueError("hvar should not be %s." % str(self._hvar))

    @property
    def genetic_variance(self):
        r"""Genetic variance.

        Returns:
            :math:`\sigma_b^2`.
        """
        return self.sigma2_b

    @property
    def environmental_variance(self):
        r"""Environmental variance.

        Returns:
            :math:`\sigma_{\epsilon}^2`.
        """
        if self._overdispersion:
            return self.sigma2_epsilon
        return self._prodlik.latent_variance

    @property
    def heritability(self):
        r"""Narrow-sense heritability.

        Returns:
            :math:`\sigma_b^2/(\sigma_a^2+\sigma_b^2+\sigma_{\epsilon}^2)`.
        """
        total = self.genetic_variance + self.covariates_variance
        total += self.environmental_variance
        return self.genetic_variance / total

    def copy(self):
        # pylint: disable=W0212
        ep = ExpFamEP.__new__(ExpFamEP)
        self._copy_to(ep)

        ep._machine = LikNormMachine(500)
        ep._prodlik = self._prodlik

        ep._phenotype = self._phenotype
        ep._tbeta = self._tbeta.copy()
        ep.delta = self.delta
        self.v = self.v

        return ep


def _initialize(prodlik, covariates, QS):
    y = prodlik.to_normal()
    flmm = FastLMM(y, covariates, QS)
    flmm.learn(progress=False)
    gv = flmm.genetic_variance
    nv = flmm.environmental_variance
    h2 = gv / (gv + nv)
    return clip(h2, 0.01, 0.9), flmm.m
