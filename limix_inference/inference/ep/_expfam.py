from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import clip, full
from numpy.linalg import lstsq

from ...inference import FastLMM
from .liknorm import create_liknorm
from .ep import EP

class ExpFamEP(EP):
    def __init__(self, phenotype, covariates, Q0, Q1, S0):
        super(ExpFamEP, self).__init__(covariates, Q0, S0, True)
        self._logger = logging.getLogger(__name__)

        self._Q1 = Q1
        self._liknorm = create_liknorm(phenotype.likelihood_name, 350)

        h2, m = _initialize(phenotype, covariates, Q0, Q1, S0)

        n = phenotype.sample_size

        self._phenotype = phenotype
        self._tbeta = lstsq(self._tM, full(n, m))[0]
        self.delta = 1 - h2
        self.v = 1.

    def _tilted_params(self):
        ctau = self._cav_tau
        ceta = self._cav_eta
        lmom0 = self._loghz
        self._liknorm.moments(self._phenotype, ceta, ctau, lmom0, self._hmu,
                              self._hvar)


    @property
    def genetic_variance(self):
        r"""Returns :math:`\sigma_b^2`."""
        return self.sigma2_b

    @property
    def environmental_variance(self):
        r"""Returns :math:`\sigma_{\epsilon}^2`."""
        return self.sigma2_epsilon

    @property
    def heritability(self):
        r"""Returns
:math:`\sigma_b^2/(\sigma_a^2+\sigma_b^2+\sigma_{\epsilon}^2)`."""
        total = self.genetic_variance + self.covariates_variance
        total += self.environmental_variance
        return self.genetic_variance / total

def _initialize(phenotype, covariates, Q0, Q1, S0):
    y = phenotype.to_normal()
    flmm = FastLMM(y, Q0, Q1, S0, covariates=covariates)
    flmm.learn()
    gv = flmm.genetic_variance
    nv = flmm.environmental_variance
    h2 = gv / (gv + nv)
    return clip(h2, 0.01, 0.9), flmm.m
