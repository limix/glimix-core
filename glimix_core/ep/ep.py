from __future__ import absolute_import, division, unicode_literals

import logging
from math import fsum

from numpy import dot, empty, inf, isfinite, log, maximum, zeros, sqrt
from numpy.linalg import norm
from numpy_sugar import epsilon
from numpy_sugar.linalg import cho_solve, ddot, dotd

from .posterior import Posterior, PosteriorLinearKernel
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

    def __init__(self, posterior_type=Posterior):
        self._logger = logging.getLogger(__name__)

        self._posterior_type = posterior_type

        self._site = None
        self._psite = None

        self._cav = None
        self._posterior = None

        self._moments = {'log_zeroth': None, 'mean': None, 'variance': None}

        self._need_params_update = True

    def _compute_moments(self):
        raise NotImplementedError

    def _initialize(self, mean, cov):
        self._logger.debug("EP parameters initialization.")
        self._need_params_update = True

        nsamples = len(mean)

        if self._site is None:
            self._site = Site(nsamples)
            self._psite = Site(nsamples)

            self._cav = dict(tau=zeros(nsamples), eta=zeros(nsamples))

            self._posterior = self._posterior_type(self._site)

        self._posterior.set_prior_mean(mean)
        self._posterior.set_prior_cov(cov)

        if self._moments['log_zeroth'] is None:
            self._moments = {
                'log_zeroth': empty(nsamples),
                'mean': empty(nsamples),
                'variance': empty(nsamples)
            }

            self._posterior.initialize()

    def _lml(self):
        if isinstance(self._posterior, PosteriorLinearKernel):
            return self._lml_linear()

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

    def _lml_linear(self):
        L = self._posterior.L()
        cov = self._posterior.prior_cov()
        Q = cov['QS'][0][0]
        S = cov['QS'][1]
        ttau = self._site.tau
        teta = self._site.eta
        ctau = self._cav['tau']
        ceta = self._cav['eta']
        m = self._posterior.prior_mean()

        TS = ttau + ctau

        s = cov['scale']
        d = cov['delta']
        A = self._posterior._A
        tQ = sqrt(1 - d) * Q

        lml = [
            -log(L.diagonal()).sum(), #
            -0.5 * sum(log(s * S)), #
            +0.5 * sum(log(A)), #
            # lml += 0.5 * sum(log(ttau)),
            +0.5 * dot(teta * A, dot(tQ, cho_solve(L, dot(tQ.T, teta * A)))), #!=
            -0.5 * dot(teta, teta / TS), #
            +dot(m, A * teta) - 0.5 * dot(m, A * ttau * m), #
            -0.5 *
            dot(m * A * ttau, dot(tQ, cho_solve(L, dot(tQ.T, 2 * A * teta - A * ttau * m)))), #
            +sum(self._moments['log_zeroth']), #
            +0.5 * sum(log(TS)), #
            # lml -= 0.5 * sum(log(ttau)),
            -0.5 * sum(log(ctau)), #
            +0.5 * dot(ceta / TS, ttau * ceta / ctau - 2 * teta), #
            0.5 * s * d * sum(teta * A * teta)
        ]
        lml = fsum(lml)

        if not isfinite(lml):
            raise ValueError("LML should not be %f." % lml)

        return lml

    def _lml_derivative_over_mean(self, dm):
        if isinstance(self._posterior, PosteriorLinearKernel):
            return self._lml_derivative_over_mean_linear(dm)

        L = self._posterior.L()
        Q, _ = self._posterior.prior_cov()
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.prior_mean()

        dlml = dot(diff, dm)
        dlml -= dot(diff, dot(Q, cho_solve(L, dot(Q.T, (ttau * dm.T).T))))

        return dlml

    def _lml_derivative_over_mean_linear(self, dm):
        L = self._posterior.L()
        cov = self._posterior.prior_cov()
        ttau = self._site.tau
        teta = self._site.eta
        A = self._posterior._A

        Q = cov['QS'][0][0] * sqrt(1 - cov['delta'])

        di = teta - ttau * self._posterior.prior_mean()

        dlml = dot(di, ldot(A, dm))
        dlml -= dot(di * A, dot(Q, cho_solve(L, dot(Q.T, ldot(A, (ttau * dm.T).T)))))

        return dlml

    def _lml_derivative_over_cov(self, dQS):
        L = self._posterior.L()
        Q, _ = self._posterior.prior_cov()
        ttau = self._site.tau
        teta = self._site.eta

        diff = teta - ttau * self._posterior.prior_mean()

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

    def _lml_derivative_over_cov_scale(self):
        L = self._posterior.L()
        cov = self._posterior.prior_cov()
        T = self._site.tau
        A = self._posterior._A

        S = cov['QS'][1]
        d = cov['delta']
        Q = sqrt(1 - d) * cov['QS'][0][0]

        e_m = self._site.eta - T * self._posterior.prior_mean()
        Ae_m = A * e_m
        QTe_m = dot(Q.T, e_m)
        QS = dotr(Q, S)
        TA = T * A

        tQStQTdi = dot(QS, QTe_m)
        QTAe_m = dot(Q.T, Ae_m)

        dKAd_m = dot(QS, QTAe_m) + d * Ae_m

        QLQAd_m = dot(Q, cho_solve(L, QTAe_m))
        TAQLQAd_m = TA * QLQAd_m

        dlml = 0.5 * dot(Ae_m, dKAd_m)
        dlml -= sum(TAQLQAd_m * dKAd_m)
        dlml += 0.5 * dot(TAQLQAd_m, dot(QS, dot(Q.T, TAQLQAd_m)) + d * TAQLQAd_m)

        dlml -= 0.5 * dotd(ldot(TA, Q), QS.T).sum()
        dlml -= 0.5 * sum(TA * d)

        t0 = dot(cho_solve(L, dot(Q.T, ldot(TA, Q))), QS.T)
        dlml += 0.5 * dotd(ldot(TA, Q), t0).sum()

        dlml += 0.5 * d * dotd(ldot(TA, Q), cho_solve(L, dotr(Q.T, TA))).sum()

        return dlml

    def _lml_derivative_over_cov_delta(self):
        L = self._posterior.L()
        cov = self._posterior.prior_cov()
        T = self._site.tau
        A = self._posterior._A

        S = cov['QS'][1]
        d = cov['delta']
        s = cov['scale']
        Q = sqrt(1 - d) * cov['QS'][0][0]

        e_m = self._site.eta - T * self._posterior.prior_mean()
        Ae_m = A * e_m
        QTe_m = dot(Q.T, e_m)
        QS = dotr(Q, S)
        TA = T * A

        tQStQTdi = dot(QS, QTe_m)
        QTAe_m = dot(Q.T, Ae_m)

        dKAd_m = - s * dot(QS, QTAe_m)/(1 - d) + s * Ae_m

        QLQAd_m = dot(Q, cho_solve(L, QTAe_m))
        TAQLQAd_m = TA * QLQAd_m

        dlml = 0.5 * dot(Ae_m, dKAd_m)
        dlml -= sum(TAQLQAd_m * dKAd_m)
        dlml += 0.5 * dot(TAQLQAd_m, - s * dot(QS, dot(Q.T, TAQLQAd_m)) / (1 - d) + s * TAQLQAd_m)

        dlml += 0.5 * s * dotd(ldot(TA, Q), QS.T).sum() / (1 - d)
        dlml -= 0.5 * sum(TA * s)

        t0 = dot(cho_solve(L, dot(Q.T, ldot(TA, Q))), QS.T)
        dlml -= 0.5 * s * dotd(ldot(TA, Q), t0).sum() / (1 - d)

        dlml += 0.5 * s * dotd(ldot(TA, Q), cho_solve(L, dotr(Q.T, TA))).sum()

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
            msg += " Last EP step was: %.10f." % norm(self._site.tau -
                                                      self._psite.tau)
            raise ValueError(msg)

        self._need_params_update = False
        self._logger.debug('EP loop has performed %d iterations.', i)
