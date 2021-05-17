from copy import copy

from numpy import asarray, ascontiguousarray, clip, dot, exp, log, zeros
from optimix import Function, Scalar, Vector

from .._util import (
    check_covariates,
    check_economic_qs,
    check_outcome,
    economic_qs_zeros,
)


class GLMM(Function):
    r"""Generalised Linear Gaussian Processes.

    The variances :math:`v_0` and :math:`v_1` are internally replaced by
    the scale and ratio parameters:

    .. math::
        v_0 = s (1 - \delta) ~\text{ and }~
        v_1 = s \delta.

    Let

    .. math::

        \mathrm Q \mathrm E \mathrm Q^{\intercal}
        = \mathrm K = \mathrm G\mathrm G^{\intercal}

    be the eigen decomposition of the random effect's covariance.
    It turns out that the covariance of the prior can be described as

    .. math::

        s \mathrm Q ((1-\delta)\mathrm E
        + \delta\mathrm I) \mathrm Q^{\intercal}.

    This means that :math:`\mathrm Q` does not change during inference, and
    this fact is used in the implementation to speed-up inference.

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
        Economic eigen decomposition.
    """

    def __init__(self, y, lik, X, QS=None):
        y = ascontiguousarray(y, float)
        X = asarray(X, float)

        Function.__init__(
            self,
            "GLMM",
            beta=Vector(zeros(X.shape[1])),
            logscale=Scalar(0.0),
            logitdelta=Scalar(0.0),
        )

        if not isinstance(lik, (tuple, list)):
            lik = (lik,)

        self._lik = (lik[0].lower(),) + tuple(ascontiguousarray(i) for i in lik[1:])
        self._y = check_outcome(y, self._lik)
        self._X = check_covariates(X)
        if QS is None:
            self._QS = economic_qs_zeros(self._y.shape[0])
        else:
            self._QS = check_economic_qs(QS)
            if self._y.shape[0] != self._QS[0][0].shape[0]:
                raise ValueError("Number of samples in outcome and covariance differ.")

        if self._y.shape[0] != self._X.shape[0]:
            raise ValueError("Number of samples in outcome and covariates differ.")

        self._factr = 1e5
        self._pgtol = 1e-6
        self._verbose = False
        self.set_variable_bounds("logscale", (log(0.001), 6.0))

        self.set_variable_bounds("logitdelta", (-50, +15))

        if lik[0] == "probit":
            self.delta = 0.0
            self.fix("delta")

    def _copy_to(self, to):
        d = to._variables
        s = self._variables

        d.get("beta").value = asarray(s.get("beta").value, float)
        d.get("beta").bounds = s.get("beta").bounds

        for v in ["logscale", "logitdelta"]:
            d.get(v).value = float(s.get(v).value)
            d.get(v).bounds = s.get(v).bounds

    @property
    def beta(self):
        r"""Fixed-effect sizes.

        Returns
        -------
        :class:`numpy.ndarray`
            :math:`\boldsymbol\beta`.
        """
        return asarray(self._variables.get("beta").value, float)

    @beta.setter
    def beta(self, v):
        self._variables.get("beta").value = v

    def copy(self):
        r"""Create a copy of this object."""
        return copy(self)

    def covariance(self):
        r"""Covariance of the prior.

        Returns
        -------
        :class:`numpy.ndarray`
            :math:`v_0 \mathrm K + v_1 \mathrm I`.
        """
        from numpy_sugar.linalg import ddot, sum2diag

        Q0 = self._QS[0][0]
        S0 = self._QS[1]
        return sum2diag(dot(ddot(Q0, self.v0 * S0), Q0.T), self.v1)

    @property
    def delta(self):
        r"""Get or set the ratio of variance between ``K`` and ``I``.

        Returns
        -------
        float
            :math:`\delta`.
        """
        return 1 / (1 + exp(-self.logitdelta))

    @delta.setter
    def delta(self, v):
        from numpy_sugar import epsilon

        v = clip(v, epsilon.small, 1 - epsilon.small)
        self.logitdelta = log(v / (1 - v))

    def fix(self, var_name):
        r"""Prevent a variable to be adjusted.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        Function._fix(self, _to_internal_name(var_name))

    def fit(self, verbose=True, factr=1e5, pgtol=1e-7):
        r"""Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        factr : float, optional
            The iteration stops when
            ``(f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps``, where ``eps`` is
            the machine precision.
        pgtol : float, optional
            The iteration will stop when ``max{|proj g_i | i = 1, ..., n} <= pgtol``
            where ``pg_i`` is the i-th component of the projected gradient.

        Notes
        -----
        Please, refer to :func:`scipy.optimize.fmin_l_bfgs_b` for further information
        about ``factr`` and ``pgtol``.
        """
        self._verbose = verbose
        self._maximize(verbose=verbose, factr=factr, pgtol=pgtol)
        self._verbose = False

    def lml(self):
        r"""Log of the marginal likelihood.

        Returns
        -------
        float
            :math:`\log p(\mathbf y)`
        """
        return self.value()

    @property
    def logitdelta(self):
        return float(self._variables.get("logitdelta").value)

    @logitdelta.setter
    def logitdelta(self, v):
        self._variables.get("logitdelta").value = v

    @property
    def logscale(self):
        return float(self._variables.get("logscale").value)

    @logscale.setter
    def logscale(self, v):
        self._variables.get("logscale").value = v

    def mean(self):
        r"""Mean of the prior.

        Returns
        -------
        :class:`numpy.ndarray`
            :math:`\mathrm X\boldsymbol\beta`.
        """
        return dot(self._X, self.beta)

    @property
    def scale(self):
        r"""Get or set the overall variance.

        Returns
        -------
        float
            :math:`s`.
        """
        return exp(self.logscale)

    @scale.setter
    def scale(self, v):
        b = self._variables.get("logscale").bounds
        self.logscale = clip(log(v), b[0], b[1])

    def set_variable_bounds(self, var_name, bounds):
        self._variables.get(var_name).bounds = bounds

    def unfix(self, var_name):
        r"""Let a variable be adjusted.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        Function._unfix(self, _to_internal_name(var_name))

    @property
    def v0(self):
        r"""First variance.

        Returns
        -------
        float
            :math:`v_0 = s (1 - \delta)`
        """
        return self.scale * (1 - self.delta)

    @property
    def v1(self):
        r"""Second variance.

        Returns
        -------
        float
            :math:`v_1 = s \delta`
        """
        return self.scale * self.delta

    def value(self, *_):
        raise NotImplementedError

    def gradient(self, *_):
        raise NotImplementedError

    def mean_star(self, Xstar):
        return dot(Xstar, self.beta)

    def variance_star(self, kss):
        return kss * self.v0 + self.v1

    def covariance_star(self, ks):
        return ks * self.v0


def _to_internal_name(name):
    translation = dict(scale="logscale", delta="logitdelta", beta="beta")
    return translation[name]
