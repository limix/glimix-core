from __future__ import absolute_import, division, unicode_literals

import logging
from math import fsum

from numpy import dot, empty, inf, isfinite, log, maximum, sqrt, zeros
from numpy.linalg import norm
from numpy_sugar import epsilon
from numpy_sugar.linalg import cho_solve, ddot, dotd

from .posterior import Posterior
from .site import Site

MAX_ITERS = 100
RTOL = epsilon.small * 1000
ATOL = epsilon.small * 1000


def ldot(A, B):
    return ddot(A, B, left=True)


def dotr(A, B):
    return ddot(A, B, left=False)


class EP(object):  # pylint: disable=R0903
    r"""Expectation Propagation algorithm.

    Let

    .. math::

        \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K)

    be the prior distribution.
    This class estimates the log of the marginal likelihood

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | \mu_i = g(z_i))
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    via Expectation Propagation and provides its gradient.

    Attributes:
        _site (:class:`glimix_core.ep.site.Site`): site-likelihood.
        _psite (:class:`glimix_core.ep.site.Site`): previous
                                                        site-likelihood.
        _cav (:class:`glimix_core.ep.cavity.Cavity`): cavity distribution.
        _posterior (:class:`glimix_core.ep.posterior.Posterior`):
                                                        posterior distribution.
        _moments (dict): moments for KL moment matching.
    """

    def __init__(self, nsites, posterior_type=Posterior):
        self._logger = logging.getLogger(__name__)

        self._posterior_type = Posterior

        self._site = Site(nsites)
        self._psite = Site(nsites)

        self._cav = dict(tau=zeros(nsites), eta=zeros(nsites))
        self._posterior = posterior_type(self._site)

        self._moments = {
            'log_zeroth': empty(nsites),
            'mean': empty(nsites),
            'variance': empty(nsites)
        }

        self._need_params_update = True

    def _compute_moments(self):
        r"""Compute zero-th, first, and second moments.

        This has to be implemented by a parent class.
        """
        raise NotImplementedError

    def _set_prior(self, mean, cov):
        self._logger.debug("Setting EP prior.")
        self._posterior.mean = mean
        self._posterior.cov = cov
        self._need_params_update = True

    def _lml(self):
        self._params_update()

        L = self._posterior.L()
        Q, S = self._posterior.cov['QS']
        Q = Q[0]
        ttau = self._site.tau
        teta = self._site.eta
        ctau = self._cav['tau']
        ceta = self._cav['eta']
        m = self._posterior.mean

        TS = ttau + ctau

        lml = [
            -log(L.diagonal()).sum(),
            -0.5 * sum(log(S)),
            # lml += 0.5 * sum(log(ttau)),
            +0.5 * dot(teta, dot(Q, cho_solve(L, dot(Q.T, teta)))),
            -0.5 * dot(teta, teta / TS),
            +dot(m, teta) - 0.5 * dot(m, ttau * m),
            -0.5 * dot(m * ttau,
                       dot(Q, cho_solve(L, dot(Q.T, 2 * teta - ttau * m)))),
            +sum(self._moments['log_zeroth']),
            +0.5 * sum(log(TS)),
            # lml -= 0.5 * sum(log(ttau)),
            -0.5 * sum(log(ctau)),
            +0.5 * dot(ceta / TS, ttau * ceta / ctau - 2 * teta)
        ]
        lml = fsum(lml)

        if not isfinite(lml):
            raise ValueError("LML should not be %f." % lml)

        return lml

    def _lml_derivative_over_mean(self, dm):
        self._params_update()

        L = self._posterior.L()
        Q = self._posterior.cov['QS'][0][0]
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.mean

        dlml = dot(diff, dm)
        dlml -= dot(diff, dot(Q, cho_solve(L, dot(Q.T, (ttau * dm.T).T))))

        return dlml

    def _lml_derivative_over_cov(self, dQS):
        self._params_update()

        L = self._posterior.L()
        Q = self._posterior.cov['QS'][0][0]
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.mean

        v0 = dot(dQS[0][0], dQS[1] * dot(dQS[0][0].T, diff))
        v1 = ttau * dot(Q, cho_solve(L, dot(Q.T, diff)))
        dlml = 0.5 * dot(diff, v0)
        dlml -= dot(v0, v1)
        dlml += 0.5 * dot(v1, dot(dQS[0][0], dQS[1] * dot(dQS[0][0].T, v1)))
        dqs = ddot(dQS[1], dQS[0][0].T, left=True)
        diag = dotd(dQS[0][0], dqs)
        dlml -= 0.5 * sum(ttau * diag)

        tmp = cho_solve(L, dot(ddot(Q.T, ttau, left=False), dQS[0][0]))
        dlml += 0.5 * sum(ttau * dotd(Q, dot(tmp, dqs)))

        return dlml

    def _params_update(self):
        if not self._need_params_update:
            return

        self._logger.debug('EP parameters update loop has started.')
        self._posterior.update()

        i = 0
        tol = -inf
        while i < MAX_ITERS and norm(self._site.tau - self._psite.tau) > tol:
            self._psite.tau[:] = self._site.tau
            self._psite.eta[:] = self._site.eta

            self._cav['tau'][:] = maximum(self._posterior.tau - self._site.tau,
                                          0)
            self._cav['eta'][:] = self._posterior.eta - self._site.eta
            self._compute_moments()

            self._site.update(self._moments['mean'], self._moments['variance'],
                              self._cav)

            self._posterior.update()

            tol = RTOL * norm(self._psite.tau) + ATOL
            i += 1

        if i == MAX_ITERS:
            msg = ('Maximum number of EP iterations has' + ' been attained.')
            msg += " Last EP step was: %.10f." % norm(
                self._site.tau - self._psite.tau)
            raise ValueError(msg)

        self._need_params_update = False
        self._logger.debug('EP loop has performed %d iterations.', i)
