from __future__ import absolute_import, division, unicode_literals

import logging
from math import fsum

from numpy import dot, empty, inf, isfinite, log, maximum, zeros
from numpy.linalg import norm
from numpy_sugar import epsilon
from numpy_sugar.linalg import cho_solve, ddot, dotd

from .posterior import Posterior
from .site import Site

MAX_ITERS = 30
RTOL = epsilon.small
ATOL = epsilon.small


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
        _site (:class:`limix_inference.ep.site.Site`): site-likelihood.
        _psite (:class:`limix_inference.ep.site.Site`): previous
                                                        site-likelihood.
        _cav (:class:`limix_inference.ep.cavity.Cavity`): cavity distribution.
        _posterior (:class:`limix_inference.ep.posterior.Posterior`):
                                                        posterior distribution.
        _moments (dict): moments for KL moment matching.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)

        self._site = None
        self._psite = None

        self._cav = None
        self._posterior = None

        self._moments = {'log_zeroth': None, 'mean': None, 'variance': None}

    def _compute_moments(self):
        raise NotImplementedError

    def _initialize(self, mean, QS):
        self._logger.debug("EP parameters initialization.")

        nsamples = len(mean)

        self._site = Site(nsamples)
        self._psite = Site(nsamples)

        self._cav = dict(tau=zeros(nsamples), eta=zeros(nsamples))
        self._posterior = Posterior(self._site)
        self._posterior.set_prior_mean(mean)
        self._posterior.set_prior_cov(QS)

        self._moments = {
            'log_zeroth': empty(nsamples),
            'mean': empty(nsamples),
            'variance': empty(nsamples)
        }

        self._posterior.initialize()

    def _lml(self):
        L = self._posterior.L()
        Q, S = self._posterior.prior_cov()
        ttau = self._site.tau
        teta = self._site.eta
        ctau = self._cav['tau']
        ceta = self._cav['eta']
        m = self._posterior.prior_mean()

        TS = ttau + ctau

        lml = [
            -log(L.diagonal()).sum(),
            -0.5 * sum(log(S)),
            # lml += 0.5 * sum(log(ttau)),
            +0.5 * dot(teta, dot(Q, cho_solve(L, dot(Q.T, teta)))),
            -0.5 * dot(teta, teta / TS),
            +dot(m, teta) - 0.5 * dot(m, ttau * m),
            -0.5 *
            dot(m * ttau, dot(Q, cho_solve(L, dot(Q.T, 2 * teta - ttau * m)))),
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
        L = self._posterior.L()
        Q, _ = self._posterior.prior_cov()
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.prior_mean()

        dlml = dot(diff, dm)
        dlml -= dot(diff, dot(Q, cho_solve(L, dot(Q.T, (ttau*dm.T).T))))

        return dlml

    def _lml_derivative_over_cov(self, dQS):
        L = self._posterior.L()
        Q, _ = self._posterior.prior_cov()
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.prior_mean()

        v0 = dot(dQS[0], dQS[1] * dot(dQS[0].T, diff))
        v1 = ttau * dot(Q, cho_solve(L, dot(Q.T, diff)))
        dlml = 0.5 * dot(diff, v0)
        dlml -= dot(v0, v1)
        dlml += 0.5 * dot(v1, dot(dQS[0], dQS[1] * dot(dQS[0].T, v1)))
        dqs = ddot(dQS[1], dQS[0].T, left=True)
        diag = dotd(dQS[0], dqs)
        dlml -= 0.5 * sum(ttau * diag)

        tmp = cho_solve(L, dot(ddot(Q.T, ttau, left=False), dQS[0]))
        dlml += 0.5 * sum(ttau * dotd(Q, dot(tmp, dqs)))

        return dlml

    def _params_update(self):
        self._logger.debug('EP parameters update loop has started.')

        i = 0
        tol = -inf
        while i < MAX_ITERS and norm(self._site.tau - self._psite.tau) > tol:
            self._psite.tau[:] = self._site.tau
            self._psite.eta[:] = self._site.eta

            self._cav['tau'][:] = maximum(self._posterior.tau - self._site.tau, 0)
            self._cav['eta'][:] = self._posterior.eta - self._site.eta
            self._compute_moments()

            self._site.update(self._moments['mean'], self._moments['variance'],
                              self._cav)

            self._posterior.update()

            tol = RTOL * norm(self._psite.tau) + ATOL
            i += 1

        if i == MAX_ITERS:
            raise ValueError('Maximum number of EP iterations has' +
                             ' been attained.')

        self._logger.debug('EP loop has performed %d iterations.', i)
