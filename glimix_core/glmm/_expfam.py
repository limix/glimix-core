from copy import copy

from numpy import asarray, diag, dot, exp
from numpy.linalg import pinv, solve

from glimix_core._ep import EPLinearKernel

from ._glmm import GLMM


class GLMMExpFam(GLMM):
    r"""Generalised Linear Gaussian Processes implementation.

    It implements inference over GLMMs via the Expectation Propagation [Min01]_
    algorithm.
    It currently supports the ``"Bernoulli"``, ``"Probit"``, ``"Binomial"``, and
    ``"Poisson"`` likelihoods. (For heterogeneous Normal likelihood, please refer to
    :class:`glimix_core.glmm.GLMMNormal` for a closed-form inference.)

    Parameters
    ----------
    y : array_like
        Outcome variable.
    lik : tuple
        Likelihood definition. The first item is one of the following likelihood names:
        ``"Bernoulli"``, ``"Binomial"``, ``"Normal"``, and ``"Poisson"``. For
        `Binomial`, the second item is an array of outcomes.
    X : array_like
        Covariates.
    QS : tuple
        Economic eigendecomposition.

    Example
    -------

    .. doctest::

        >>> from numpy import dot, sqrt, zeros
        >>> from numpy.random import RandomState
        >>>
        >>> from numpy_sugar.linalg import economic_qs
        >>>
        >>> from glimix_core.glmm import GLMMExpFam
        >>>
        >>> random = RandomState(0)
        >>> nsamples = 10
        >>>
        >>> X = random.randn(nsamples, 2)
        >>> G = random.randn(nsamples, 100)
        >>> K = dot(G, G.T)
        >>> ntrials = random.randint(1, 100, nsamples)
        >>> z = dot(G, random.randn(100)) / sqrt(100)
        >>>
        >>> successes = zeros(len(ntrials), int)
        >>> for i in range(len(ntrials)):
        ...    successes[i] = sum(z[i] + 0.2 * random.randn(ntrials[i]) > 0)
        >>>
        >>> QS = economic_qs(K)
        >>>
        >>> glmm = GLMMExpFam(successes, ('binomial', ntrials), X, QS)
        >>> print('Before: %.2f' % glmm.lml())
        Before: -16.40
        >>> glmm.fit(verbose=False)
        >>> print('After: %.2f' % glmm.lml())
        After: -13.43
    """

    def __init__(self, y, lik, X, QS=None, n_int=1000, rtol=1.49e-05, atol=1.49e-08):
        from liknorm import LikNormMachine

        GLMM.__init__(self, y, lik, X, QS)

        self._ep = EPLinearKernel(self._X.shape[0], rtol=rtol, atol=atol)
        self._ep.set_compute_moments(self.compute_moments)
        self._machine = LikNormMachine(self._lik[0], n_int)
        self.update_approx = True

        self._variables.get("beta").listen(self.set_update_approx)
        self._variables.get("logscale").listen(self.set_update_approx)
        self._variables.get("logitdelta").listen(self.set_update_approx)

    def __copy__(self):
        gef = GLMMExpFam(self._y, self._lik, self._X, self._QS)
        gef.__dict__["_ep"] = copy(self._ep)
        gef.__dict__["_ep"].set_compute_moments(gef.compute_moments)
        gef.update_approx = self.update_approx

        GLMM._copy_to(self, gef)

        return gef

    def __del__(self):
        if hasattr(self, "_machine"):
            self._machine.finish()

    def _update_approx(self):
        if not self.update_approx:
            return

        self._ep.set_prior(self.mean(), self.covariance())
        self.update_approx = False

    @property
    def beta(self):
        return GLMM.beta.fget(self)

    @beta.setter
    def beta(self, v):
        GLMM.beta.fset(self, v)
        self.set_update_approx()

    def compute_moments(self, eta, tau, moments):
        y = (self._y,) + self._lik[1:]
        self._machine.moments(y, eta, tau, moments)

    def covariance(self):
        return dict(QS=self._QS, scale=self.scale, delta=self.delta)

    def fit(self, verbose=True, factr=1e5, pgtol=1e-7):
        self._ep.verbose = verbose
        super(GLMMExpFam, self).fit(verbose=verbose, factr=factr, pgtol=pgtol)
        self._ep.verbose = False

    def fix(self, var_name):
        GLMM.fix(self, var_name)
        self.set_update_approx()

    def posteriori_mean(self):
        r"""Mean of the estimated posteriori.

        This is also the maximum a posteriori estimation of the latent variable.
        """
        from numpy_sugar.linalg import rsolve

        Sigma = self.posteriori_covariance()
        eta = self._ep._posterior.eta
        return dot(Sigma, eta + rsolve(GLMM.covariance(self), self.mean()))

    def posteriori_covariance(self):
        r"""Covariance of the estimated posteriori."""
        K = GLMM.covariance(self)
        tau = self._ep._posterior.tau
        return pinv(pinv(K) + diag(1 / tau))

    def gradient(self):
        r"""Gradient of the log of the marginal likelihood.

        Returns
        -------
        dict
            Map between variables to their gradient values.
        """
        self._update_approx()

        g = self._ep.lml_derivatives(self._X)
        ed = exp(-self.logitdelta)
        es = exp(self.logscale)

        grad = dict()
        grad["logitdelta"] = g["delta"] * (ed / (1 + ed)) / (1 + ed)
        grad["logscale"] = g["scale"] * es
        grad["beta"] = g["mean"]

        return grad

    @property
    def logitdelta(_):
        return super().logitdelta

    @logitdelta.setter
    def logitdelta(self, v):
        GLMM.logitdelta.fset(self, v)
        self.set_update_approx()

    @property
    def logscale(_):
        return super().logscale

    @logscale.setter
    def logscale(self, v):
        GLMM.logscale.fset(self, v)
        self.set_update_approx()

    def set_update_approx(self, _=None):
        self.update_approx = True

    def set_variable_bounds(self, var_name, bounds):
        GLMM.set_variable_bounds(self, var_name, bounds)
        self.set_update_approx()

    @property
    def site(self):
        return self._ep.site

    def unfix(self, var_name):
        GLMM.unfix(self, var_name)
        self.set_update_approx()

    def value(self):
        self._update_approx()
        return self._ep.lml()

    def predictive_mean(self, Xstar, ks, kss):
        mstar = self.mean_star(Xstar)
        ks = self.covariance_star(ks)
        m = self.mean()
        eta = self._ep.posterior.eta
        tau = self._ep.posterior.tau
        mu = eta / tau
        K = GLMM.covariance(self)
        return mstar + dot(ks, solve(K, mu - m))

    def predictive_covariance(self, Xstar, ks, kss):
        from numpy_sugar.linalg import sum2diag

        kss = self.variance_star(kss)
        ks = self.covariance_star(ks)
        tau = self._ep.posterior.tau
        K = GLMM.covariance(self)
        KT = sum2diag(K, 1 / tau)
        ktk = solve(KT, ks.T)
        b = []
        for i in range(len(kss)):
            b += [dot(ks[i, :], ktk[:, i])]
        b = asarray(b)
        return kss - b
