from __future__ import absolute_import, division, unicode_literals

import logging

from numpy_sugar import is_all_finite
from numpy import ascontiguousarray, clip, full
from numpy.linalg import lstsq

from liknorm import LikNormMachine


# class GGP(object):
#     def __init__(self, prodlik, mean, cov):
#         self._logger = logging.getLogger(__name__)
#
#         self._machine = LikNormMachine(prodlik.name, 500)
#         self._prodlik = prodlik
#
#     def _tilted_params(self):
#         y = self._phenotype.ytuple
#         ctau = self._cav_tau
#         ceta = self._cav_eta
#         moments = {'log_zeroth': self._loghz, 'mean': self._hmu,
#                    'variance': self._hvar}
#         self._machine.moments(y, ceta, ctau, moments)

#     @property
#     def genetic_variance(self):
#         r"""Genetic variance.
#
#         Returns:
#             :math:`\sigma_b^2`.
#         """
#         return self.sigma2_b
#
#     @property
#     def environmental_variance(self):
#         r"""Environmental variance.
#
#         Returns:
#             :math:`\sigma_{\epsilon}^2`.
#         """
#         if self._overdispersion:
#             return self.sigma2_epsilon
#         return self._prodlik.latent_variance
#
#     @property
#     def heritability(self):
#         r"""Narrow-sense heritability.
#
#         Returns:
#             :math:`\sigma_b^2/(\sigma_a^2+\sigma_b^2+\sigma_{\epsilon}^2)`.
#         """
#         total = self.genetic_variance + self.covariates_variance
#         total += self.environmental_variance
#         return self.genetic_variance / total
#
#     def copy(self):
#         # pylint: disable=W0212
#         ep = GGP.__new__(GGP)
#         self._copy_to(ep)
#
#         ep._machine = LikNormMachine(500)
#         ep._prodlik = self._prodlik
#
#         ep._phenotype = self._phenotype
#         ep._tbeta = self._tbeta.copy()
#         ep.delta = self.delta
#         self.v = self.v
#
#         return ep
#
#
# def _initialize(prodlik, covariates, QS):
#     y = prodlik.to_normal()
#     flmm = FastLMM(y, covariates, QS)
#     flmm.learn(progress=False)
#     gv = flmm.genetic_variance
#     nv = flmm.environmental_variance
#     h2 = gv / (gv + nv)
#     return clip(h2, 0.01, 0.9), flmm.m
