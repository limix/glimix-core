from copy import deepcopy

from numpy import dot, exp, eye, log, pi, trace, zeros
from numpy.linalg import slogdet, solve

from ..lmm import FastScanner
from ._glmm import GLMM


class GLMMNormal(GLMM):
    r"""
    LMM with heterogeneous Normal likelihoods.

    We model

    .. math::

        \tilde{\boldsymbol\mu} \sim \mathcal N(\mathrm X\boldsymbol\beta,
        v_0 \mathrm K + v_1 \mathrm I + \tilde{\Sigma}),

    where :math:`\tilde{\boldsymbol\mu}` and :math:`\tilde{\Sigma}` are
    typically given by EP approximation to non-Normal likelihood (please, refer
    to :class:`glimix_core.expfam.GLMMExpFam`).
    If that is the case, :math:`\tilde{\Sigma}` is a diagonal matrix with non-negative
    values.
    Those EP parameters are given to this class via their natural forms:
    ``eta`` and ``tau``.

    Parameters
    ----------
    eta : array_like
        :math:`[\tilde{\mu}_0\tilde{\sigma}^{-2}_0 \quad \tilde{\mu}_1\tilde{\sigma}^{-2}_1 \quad\cdots\quad \tilde{\mu}_n\tilde{\sigma}^{-2}_n]`.
    tau : array_like
        :math:`[\tilde{\sigma}^{-2}_0\quad\tilde{\sigma}^{-2}_1 \quad\cdots\quad \tilde{\sigma}^{-2}_n]`.
    X : array_like
        Covariates.
    QS : tuple
        Economic eigendecomposition of :math:`\mathrm K`.
    """

    def __init__(self, eta, tau, X, QS=None):
        self._cache = dict(value=None, grad=None)
        GLMM.__init__(self, eta, ("normal", tau), X, QS)
        self._variables.get("beta").listen(self.clear_cache)
        self._variables.get("logscale").listen(self.clear_cache)
        self._variables.get("logitdelta").listen(self.clear_cache)

    def __copy__(self):
        gef = GLMMNormal(self.eta, self.tau, self._X, self._QS)
        GLMM._copy_to(self, gef)
        gef.__dict__["_cache"] = deepcopy(self._cache)
        return gef

    def clear_cache(self, _=None):
        self._cache["value"] = None
        self._cache["grad"] = None

    @property
    def beta(self):
        return GLMM.beta.fget(self)

    @beta.setter
    def beta(self, v):
        GLMM.beta.fset(self, v)
        self.clear_cache()

    @property
    def logitdelta(self):
        return GLMM.logitdelta.fget(self)

    @logitdelta.setter
    def logitdelta(self, v):
        GLMM.logitdelta.fset(self, v)
        self.clear_cache()

    @property
    def logscale(self):
        return GLMM.logscale.fget(self)

    @logscale.setter
    def logscale(self, v):
        GLMM.logscale.fset(self, v)
        self.clear_cache()

    def fix(self, var_name):
        GLMM.fix(self, var_name)
        self.clear_cache()

    @property
    def eta(self):
        return self._y

    @property
    def tau(self):
        return self._lik[1]

    def get_fast_scanner(self):
        r"""Return :class:`glimix_core.lmm.FastScanner` for the current
        delta."""
        from numpy_sugar.linalg import ddot, economic_qs, sum2diag

        y = self.eta / self.tau

        if self._QS is None:
            K = eye(y.shape[0]) / self.tau
        else:
            Q0 = self._QS[0][0]
            S0 = self._QS[1]
            K = dot(ddot(Q0, self.v0 * S0), Q0.T)
            K = sum2diag(K, 1 / self.tau)

        return FastScanner(y, self._X, economic_qs(K), self.v1)

    def gradient(self):
        from numpy_sugar.linalg import ddot, sum2diag

        if self._cache["grad"] is not None:
            return self._cache["grad"]

        scale = exp(self.logscale)
        delta = 1 / (1 + exp(-self.logitdelta))

        v0 = scale * (1 - delta)
        v1 = scale * delta

        mu = self.eta / self.tau
        n = len(mu)
        if self._QS is None:
            K = zeros((n, n))
        else:
            Q0 = self._QS[0][0]
            S0 = self._QS[1]
            K = dot(ddot(Q0, S0), Q0.T)

        A = sum2diag(sum2diag(v0 * K, v1), 1 / self.tau)
        X = self._X

        m = mu - self.mean()
        g = dict()
        Aim = solve(A, m)

        g["beta"] = dot(m, solve(A, X))

        Kd0 = sum2diag((1 - delta) * K, delta)
        Kd1 = sum2diag(-scale * K, scale)

        g["scale"] = -trace(solve(A, Kd0))
        g["scale"] += dot(Aim, dot(Kd0, Aim))
        g["scale"] *= 1 / 2

        g["delta"] = -trace(solve(A, Kd1))
        g["delta"] += dot(Aim, dot(Kd1, Aim))
        g["delta"] *= 1 / 2

        ed = exp(-self.logitdelta)
        es = exp(self.logscale)

        grad = dict()
        grad["logitdelta"] = g["delta"] * (ed / (1 + ed)) / (1 + ed)
        grad["logscale"] = g["scale"] * es
        grad["beta"] = g["beta"]

        self._cache["grad"] = grad

        return self._cache["grad"]

    def set_variable_bounds(self, var_name, bounds):
        GLMM.set_variable_bounds(self, var_name, bounds)
        self.clear_cache()

    def unfix(self, var_name):
        GLMM.unfix(self, var_name)
        self.clear_cache()

    def value(self):
        r"""Log of the marginal likelihood.

        Formally,

        .. math::

            - \frac{n}{2}\log{2\pi} - \frac{1}{2} \log{\left|
                v_0 \mathrm K + v_1 \mathrm I + \tilde{\Sigma} \right|}
                    - \frac{1}{2}
                    \left(\tilde{\boldsymbol\mu} -
                    \mathrm X\boldsymbol\beta\right)^{\intercal}
                    \left( v_0 \mathrm K + v_1 \mathrm I +
                    \tilde{\Sigma} \right)^{-1}
                    \left(\tilde{\boldsymbol\mu} -
                    \mathrm X\boldsymbol\beta\right)

        Returns
        -------
        float
            :math:`\log{p(\tilde{\boldsymbol\mu})}`
        """
        from numpy_sugar.linalg import ddot, sum2diag

        if self._cache["value"] is not None:
            return self._cache["value"]

        scale = exp(self.logscale)
        delta = 1 / (1 + exp(-self.logitdelta))

        v0 = scale * (1 - delta)
        v1 = scale * delta

        mu = self.eta / self.tau
        n = len(mu)
        if self._QS is None:
            K = zeros((n, n))
        else:
            Q0 = self._QS[0][0]
            S0 = self._QS[1]
            K = dot(ddot(Q0, S0), Q0.T)

        A = sum2diag(sum2diag(v0 * K, v1), 1 / self.tau)
        m = mu - self.mean()

        v = -n * log(2 * pi)
        v -= slogdet(A)[1]
        v -= dot(m, solve(A, m))

        self._cache["value"] = v / 2

        return self._cache["value"]
