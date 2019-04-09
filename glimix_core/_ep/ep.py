from copy import deepcopy
from math import fsum

from numpy import dot, empty, inf, isfinite, log, maximum, zeros
from numpy.linalg import norm

from .posterior import Posterior
from .site import Site

MAX_ITERS = 100


class EP(object):
    """
    Expectation Propagation algorithm.

    Let ::

        𝒩(𝐳｜𝐦, K)

    be the prior distribution.
    This class estimates the log of the marginal likelihood ::

        p(𝐲) = ∫∏ᵢp(yᵢ｜μᵢ = g(zᵢ))𝒩(𝐳｜𝐦, K) d𝐳

    via Expectation Propagation and provides its gradient.

    Attributes
    ----------
        _site: site-likelihood.
        _psite: previous site-likelihood.
        _cav: cavity distribution.
        _posterior: posterior distribution.
        _moments: moments for KL moment matching.
    """

    def __init__(self, nsites, posterior_type=Posterior, rtol=1.49e-05, atol=1.49e-08):
        self._site = Site(nsites)
        self._psite = Site(nsites)
        self._rtol = rtol
        self._atol = atol

        self._cav = dict(tau=zeros(nsites), eta=zeros(nsites))
        self._posterior = posterior_type(self._site)

        self._moments = {
            "log_zeroth": empty(nsites),
            "mean": empty(nsites),
            "variance": empty(nsites),
        }

        self._need_update = True
        self._compute_moments = None
        self._cache = dict(lml=None, grad=None)
        self.verbose = False

    def __copy__(self):
        cls = self.__class__
        ep = cls.__new__(cls)

        ep.__dict__["_site"] = deepcopy(self.site)
        ep.__dict__["_psite"] = deepcopy(self.__dict__["_psite"])

        ep.__dict__["_rtol"] = self._rtol
        ep.__dict__["_atol"] = self._atol

        ep.__dict__["_cav"] = deepcopy(self.cav)

        posterior_type = type(self.posterior)
        ep.__dict__["_posterior"] = posterior_type(ep.site)

        ep.__dict__["_moments"] = deepcopy(self.moments)

        ep.__dict__["_need_update"] = self._need_update
        ep.__dict__["_compute_moments"] = None
        ep.__dict__["_cache"] = deepcopy(self._cache)
        ep.__dict__["verbose"] = deepcopy(self.verbose)

        return ep

    def _update(self):
        if not self._need_update:
            return

        self._posterior.update()

        i = 0
        tol = -inf
        n1 = norm(self._site.tau)
        while i < MAX_ITERS and norm(self._site.tau - self._psite.tau) > tol:
            self._psite.tau[:] = self._site.tau
            self._psite.eta[:] = self._site.eta

            self._cav["tau"][:] = maximum(self._posterior.tau - self._site.tau, 0)
            self._cav["eta"][:] = self._posterior.eta - self._site.eta
            self._compute_moments(self._cav["eta"], self._cav["tau"], self._moments)

            self._site.update(
                self._moments["mean"], self._moments["variance"], self._cav
            )

            self._posterior.update()

            n0 = n1
            n1 = norm(self._site.tau)
            tol = self._rtol * min(n0, n1) + self._atol
            i += 1
            if i % 10 == 9:
                self._rtol *= 10

        if i == MAX_ITERS:
            msg = "Maximum number of EP iterations has" + " been attained."
            msg += " Last EP step was: %.10f." % norm(self._site.tau - self._psite.tau)
            raise ValueError(msg)

        if self.verbose:
            print("{} EP iterations".format(i))
        self._need_update = False

    @property
    def cav(self):
        return self._cav

    def lml(self):
        from numpy_sugar.linalg import cho_solve

        if self._cache["lml"] is not None:
            return self._cache["lml"]

        self._update()

        L = self._posterior.L()
        Q, S = self._posterior.cov["QS"]
        Q = Q[0]
        ttau = self._site.tau
        teta = self._site.eta
        ctau = self._cav["tau"]
        ceta = self._cav["eta"]
        m = self._posterior.mean

        TS = ttau + ctau

        lml = [
            -log(L.diagonal()).sum(),
            -0.5 * sum(log(S)),
            # lml += 0.5 * sum(log(ttau)),
            +0.5 * dot(teta, dot(Q, cho_solve(L, dot(Q.T, teta)))),
            -0.5 * dot(teta, teta / TS),
            +dot(m, teta) - 0.5 * dot(m, ttau * m),
            -0.5 * dot(m * ttau, dot(Q, cho_solve(L, dot(Q.T, 2 * teta - ttau * m)))),
            +sum(self._moments["log_zeroth"]),
            +0.5 * sum(log(TS)),
            # lml -= 0.5 * sum(log(ttau)),
            -0.5 * sum(log(ctau)),
            +0.5 * dot(ceta / TS, ttau * ceta / ctau - 2 * teta),
        ]
        lml = fsum(lml)

        if not isfinite(lml):
            raise ValueError("LML should not be %f." % lml)

        self._cache["lml"] = lml
        return lml

    def lml_derivative_over_cov(self, dQS):
        from numpy_sugar.linalg import cho_solve, ddot, dotd

        self._update()

        L = self._posterior.L()
        Q = self._posterior.cov["QS"][0][0]
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
        from numpy_sugar.linalg import cho_solve

        self._update()

        L = self._posterior.L()
        Q = self._posterior.cov["QS"][0][0]
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
        for k in self._cache:
            self._cache[k] = None
