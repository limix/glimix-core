from __future__ import absolute_import, division, unicode_literals

from copy import copy

from liknorm import LikNormMachine
from numpy import asarray, ascontiguousarray, dot, exp
from numpy.linalg import norm, solve

from glimix_core.ep import EPLinearKernel
from numpy_sugar import epsilon
from numpy_sugar.linalg import sum2diag

from ._glmm import GLMM


class GLMMExpFam(GLMM):
    r"""Generalised Linear Gaussian Processes implementation.

    It implements inference over the GLMM explained in Section BLA via
    the Expectation Propagation [Min01]_ algorithm.
    It currently supports the `Bernoulli`, `Probit`, `Binomial`, and `Poisson`
    likelihoods. (For heterogeneous Normal likelihood, please refer to
    :class:`glimix_core.glmm.GLMMNormal` for a closed-form inference.)

    Parameters
    ----------
    y : array_like
        Outcome variable.
    lik_name : str
        Likelihood name. It supports `Bernoulli`, `Probit`, `Binomial`, and
        `Poisson`.
    X : array_like
        Covariates.
    QS : tuple
        Economic eigen decomposition.

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
        >>> y = (successes, ntrials)
        >>>
        >>> QS = economic_qs(K)
        >>>
        >>> glmm = GLMMExpFam(y, 'binomial', X, QS)
        >>> print('Before: %.2f' % glmm.lml())
        Before: -16.40
        >>> glmm.fit(verbose=False)
        >>> print('After: %.2f' % glmm.lml())
        After: -13.43
    """

    def __init__(self,
                 y,
                 lik_name,
                 X,
                 QS=None,
                 n_int=1000,
                 rtol=epsilon.small * 1000,
                 atol=epsilon.small):
        GLMM.__init__(self, y, lik_name, X, QS)

        self._ep = EPLinearKernel(self._X.shape[0], rtol=rtol, atol=atol)
        self._ep.set_compute_moments(self.compute_moments)
        self._machine = LikNormMachine(self._lik_name, n_int)
        self.update_approx = True

        self.variables().get('beta').listen(self.set_update_approx)
        self.variables().get('logscale').listen(self.set_update_approx)
        self.variables().get('logitdelta').listen(self.set_update_approx)

    def __copy__(self):
        gef = GLMMExpFam(self._y, self._lik_name, self._X, self._QS)
        gef.__dict__['_ep'] = copy(self._ep)
        gef.__dict__['_ep'].set_compute_moments(gef.compute_moments)
        gef.update_approx = self.update_approx

        GLMM._copy_to(self, gef)

        return gef

    def __del__(self):
        if hasattr(self, '_machine'):
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
        y = self._y
        y = tuple(ascontiguousarray(i) for i in y.T)
        self._machine.moments(y, eta, tau, moments)

    def covariance(self):
        return dict(QS=self._QS, scale=self.scale, delta=self.delta)

    def fit(self, verbose=True, factr=1e5, pgtol=1e-7):
        r"""Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        """
        self._ep.verbose = verbose
        super(GLMMExpFam, self).fit(verbose=verbose, factr=factr, pgtol=pgtol)
        self._ep.verbose = False

    def fix(self, var_name):
        GLMM.fix(self, var_name)
        self.set_update_approx()

    def gradient(self):
        self._update_approx()

        g = self._ep.lml_derivatives(self._X)
        ed = exp(-self.logitdelta)
        es = exp(self.logscale)

        grad = dict()
        grad['logitdelta'] = g['delta'] * (ed / (1 + ed)) / (1 + ed)
        grad['logscale'] = g['scale'] * es
        grad['beta'] = g['mean']

        return grad

    @property
    def logitdelta(self):
        return GLMM.logitdelta.fget(self)

    @logitdelta.setter
    def logitdelta(self, v):
        GLMM.logitdelta.fset(self, v)
        self.set_update_approx()

    @property
    def logscale(self):
        return GLMM.logscale.fget(self)

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
        if self._verbose:
            b = norm(self.beta)
            d = self.delta
            s = self.scale
            print("b={: 10.6} d={: 10.6} s={: 10.6}".format(b, d, s))
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
