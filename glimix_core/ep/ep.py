from __future__ import absolute_import, division, unicode_literals

from copy import deepcopy
from math import fsum

from numpy import dot, empty, inf, isfinite, log, maximum, sqrt, zeros
from numpy.linalg import norm
from numpy_sugar import epsilon
from numpy_sugar.linalg import cho_solve, ddot, dotd

from .posterior import Posterior
from .site import Site

MAX_ITERS = 100
RTOL = epsilon.small * 1000
ATOL = epsilon.small


class EP(object):
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

        self._site = Site(nsites)
        self._psite = Site(nsites)

        self._cav = dict(tau=zeros(nsites), eta=zeros(nsites))
        self._posterior = posterior_type(self._site)

        self._moments = {
            'log_zeroth': empty(nsites),
            'mean': empty(nsites),
            'variance': empty(nsites)
        }

        self._need_update = True
        self._compute_moments = None
        self._cache = dict(lml=None, grad=None)

    def __copy__(self):
        cls = self.__class__
        ep = cls.__new__(cls)

        ep.__dict__['_site'] = deepcopy(self.site)
        ep.__dict__['_psite'] = deepcopy(self.__dict__['_psite'])
        ep.__dict__['_cav'] = deepcopy(self.cav)

        posterior_type = type(self.posterior)
        ep.__dict__['_posterior'] = posterior_type(ep.site)

        ep.__dict__['_moments'] = deepcopy(self.moments)

        ep.__dict__['_need_update'] = self._need_update
        ep.__dict__['_compute_moments'] = None
        ep.__dict__['_cache'] = deepcopy(self._cache)

        return ep

    def _update(self):
        if not self._need_update:
            return

        self._posterior.update()

        i = 0
        tol = -inf
        while i < MAX_ITERS and norm(self._site.tau - self._psite.tau) > tol:
            self._psite.tau[:] = self._site.tau
            self._psite.eta[:] = self._site.eta

            self._cav['tau'][:] = maximum(self._posterior.tau - self._site.tau,
                                          0)
            self._cav['eta'][:] = self._posterior.eta - self._site.eta
            self._compute_moments(self._cav['eta'], self._cav['tau'],
                                  self._moments)

            self._site.update(self._moments['mean'], self._moments['variance'],
                              self._cav)

            self._posterior.update()

            n0 = norm(self._psite.tau)
            n1 = norm(self._cav['tau'])
            tol = RTOL * sqrt(n0 * n1) + ATOL
            i += 1

        if i == MAX_ITERS:
            msg = ('Maximum number of EP iterations has' + ' been attained.')
            msg += " Last EP step was: %.10f." % norm(
                self._site.tau - self._psite.tau)
            raise ValueError(msg)

        self._need_update = False

    @property
    def cav(self):
        return self._cav

    def lml(self):
        if self._cache['lml'] is not None:
            return self._cache['lml']

        self._update()

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

        self._cache['lml'] = lml
        return lml

    def lml_derivative_over_cov(self, dQS):
        self._update()

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

    def lml_derivative_over_mean(self, dm):
        self._update()

        L = self._posterior.L()
        Q = self._posterior.cov['QS'][0][0]
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.mean

        dlml = dot(diff, dm)
        dlml -= dot(diff, dot(Q, cho_solve(L, dot(Q.T, (ttau * dm.T).T))))

        return dlml

    @property
    def moments(self):
        return self._moments

    @property
    def posterior(self):
        return self._posterior

    def set_compute_moments(self, cm):
        self._compute_moments = cm

    @property
    def site(self):
        return self._site

    def set_prior(self, mean, covariance):
        self._posterior.mean = mean
        self._posterior.cov = covariance
        self._need_update = True
        self._cache['lml'] = None
        self._cache['grad'] = None
