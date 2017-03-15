from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import sum as npsum
from numpy import dot, empty, inf, isfinite, log, maximum
from numpy.linalg import norm
from numpy_sugar import epsilon
from numpy_sugar.linalg import cho_solve, ddot, dotd

from .cavity import Cavity
from .posterior import Posterior
from .site import Site

MAX_ITERS = 30
RTOL = epsilon.small
ATOL = epsilon.small


class EP(object): # pylint: disable=R0903
    r"""Expectation Propagation algorithm.

    Let

    .. math::

        \mathcal N(\mathbf z ~|~ \mathbf m; \mathrm K)

    be the prior distribution.
    This class estimates the log of the marginal likelihood

    .. math::

        p(\mathbf y) = \int \prod_i p(y_i | g(\mathrm E[y_i | z_i])=z_i)
            \mathcal N(\mathbf z ~|~ \mathbf m, \mathrm K) \mathrm d\mathbf z

    via Expectation Propagation.

    Attributes:
        _site (:class:`.site.Site`): bla.
        _psite (:class:`.site.Site`): bla.
        _cav (:class:`.site.Cavity`): bla.
        _posterior (:class:`.site.Posterior`): bla.
        _moments (dict): bla.
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

    def _initialize(self, mean, cov):
        self._logger.debug("EP parameters initialization.")

        nsamples = len(mean)

        self._site = Site(nsamples)
        self._psite = Site(nsamples)

        self._cav = Cavity(nsamples)
        self._posterior = Posterior(self._site)
        self._posterior.set_prior_mean(mean)
        self._posterior.set_prior_cov(cov)

        self._moments = {
            'log_zeroth': empty(nsamples),
            'mean': empty(nsamples),
            'variance': empty(nsamples)
        }

        self._posterior.initialize()

    def _lml(self):
        L = self._posterior.L()
        Q, S = self._posterior.QS()
        ttau = self._site.tau
        teta = self._site.eta
        ctau = self._cav.tau
        ceta = self._cav.eta
        m = self._posterior.prior_mean()

        TS = ttau + ctau

        lml = -log(L.diagonal()).sum()
        lml -= 0.5 * npsum(log(S))
        lml += 0.5 * npsum(log(ttau))
        lml += 0.5 * dot(teta, dot(Q, cho_solve(L, dot(Q.T, teta))))
        lml -= 0.5 * dot(teta, teta / TS)
        lml += dot(m, teta) - 0.5 * dot(m, ttau * m)
        lml += -0.5 * dot(m * ttau,
                          dot(Q, cho_solve(L, dot(Q.T, 2 * teta - ttau * m))))
        lml += npsum(self._moments['log_zeroth'])
        lml += 0.5 * npsum(log(TS)) - 0.5 * npsum(log(ttau))
        lml -= 0.5 * npsum(log(ctau))
        lml += 0.5 * dot(ceta / TS, ttau * ceta / ctau - 2 * teta)

        if not isfinite(lml):
            raise ValueError("LML should not be %f." % lml)

        return lml

    def _lml_derivative_over_mean(self, dm):
        L = self._posterior.L()
        Q, _ = self._posterior.QS()
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.prior_mean()

        dlml = dot(diff, dm)
        dlml -= dot(diff, dot(Q, cho_solve(L, dot(Q.T, ttau * dm))))

        return dlml

    def _lml_derivative_over_cov(self, dcov):
        L = self._posterior.L()
        Q, _ = self._posterior.QS()
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.prior_mean()

        v0 = dot(dcov, diff)
        v1 = ttau * dot(Q, cho_solve(L, dot(Q.T, diff)))

        dlml = 0.5 * dot(diff, v0)
        dlml -= dot(v0, v1)
        dlml += 0.5 * dot(v1, dot(dcov, v1))
        dlml -= 0.5 * sum(ttau * dcov.diagonal())
        TK = ddot(ttau, dcov, left=True)
        dlml += 0.5 * sum(ttau * dotd(Q, cho_solve(L, dot(Q.T, TK))))

        return dlml

    def _params_update(self):
        self._logger.debug('EP parameters update loop has started.')

        i = 0
        tol = -inf
        while i < MAX_ITERS and norm(self._site.tau - self._psite.tau) > tol:
            self._psite.tau[:] = self._site.tau
            self._psite.eta[:] = self._site.eta

            self._cav.tau[:] = maximum(self._posterior.tau - self._site.tau, 0)
            self._cav.eta[:] = self._posterior.eta - self._site.eta
            self._compute_moments()

            self._site.update(self._moments['mean'], self._moments['variance'],
                              self._cav.eta, self._cav.tau)

            self._posterior.update()

            tol = RTOL * norm(self._psite.tau) + ATOL
            i += 1

        if i == MAX_ITERS:
            self._logger.warning('Maximum number of EP iterations has' +
                                 ' been attained.')

        self._logger.debug('EP loop has performed %d iterations.', i)
