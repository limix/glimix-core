from __future__ import absolute_import, division, unicode_literals

from copy import copy

from numpy import asarray, clip, dot, exp, finfo, log, zeros
from numpy_sugar import epsilon
from numpy_sugar.linalg import ddot, sum2diag
from optimix import Function, Scalar, Vector

from ..util import check_covariates, check_economic_qs, check_outcome
from ..util import normalise_outcome


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
    lik_name : str
        Likelihood name. It supports `Bernoulli`, `Binomial`, `Normal`, and
        `Poisson`.
    X : array_like
        Covariates.
    QS : tuple
        Economic eigen decomposition.
    """

    def __init__(self, y, lik_name, X, QS):
        y = normalise_outcome(y)
        X = asarray(X, float)

        Function.__init__(
            self,
            beta=Vector(zeros(X.shape[1])),
            logscale=Scalar(0.0),
            logitdelta=Scalar(0.0))

        self._lik_name = lik_name.lower()
        self._y = check_outcome(y, self._lik_name)
        self._X = check_covariates(X)
        self._QS = check_economic_qs(QS)

        if self._y.shape[0] != self._X.shape[0]:
            raise ValueError("Number of samples in " +
                             "outcome and covariates differ.")

        if self._y.shape[0] != self._QS[0][0].shape[0]:
            raise ValueError("Number of samples in " +
                             "outcome and covariance differ.")

        self._factr = 1e5
        self._pgtol = 1e-6
        self.set_variable_bounds('logscale', (log(1e-3), 7.))
        logmax = log(finfo(float).max)
        self.set_variable_bounds('logitdelta', (-logmax, +logmax))

        self.set_nodata()

    def _copy_to(self, to):
        d = to.variables()
        s = self.variables()

        d.get('beta').value = asarray(s.get('beta').value, float)
        d.get('beta').bounds = s.get('beta').bounds

        for v in ['logscale', 'logitdelta']:
            d.get(v).value = float(s.get(v).value)
            d.get(v).bounds = s.get(v).bounds

    @property
    def beta(self):
        r"""Fixed-effect sizes.

        Returns
        -------
        array_like
            :math:`\boldsymbol\beta`.
        """
        return asarray(self.variables().get('beta').value, float)

    @beta.setter
    def beta(self, v):
        self.variables().get('beta').value = v

    def copy(self):
        r"""Create a copy of this object."""
        return copy(self)

    def covariance(self):
        r"""Covariance of the prior.

        Returns
        -------
        array_like
            :math:`v_0 \mathrm K + v_1 \mathrm I`.
        """
        Q0 = self._QS[0][0]
        S0 = self._QS[1]
        return sum2diag(dot(ddot(Q0, self.v0 * S0), Q0.T), self.v1)

    @property
    def delta(self):
        r"""Ratio of variance between ``K`` and ``I``.

        Returns
        -------
        float
            :math:`\delta`.
        """
        return 1 / (1 + exp(-self.logitdelta))

    @delta.setter
    def delta(self, v):
        v = clip(v, epsilon.large, 1 - epsilon.large)
        self.logitdelta = log(v / (1 - v))

    def fix(self, var_name):
        r"""Prevent a variable to be adjusted.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        Function.fix(self, _to_internal_name(var_name))

    def fit(self, verbose=True):
        r"""Maximise the marginal likelihood.

        Parameters
        ----------
        verbose : bool
            ``True`` for progress output; ``False`` otherwise.
            Defaults to ``True``.
        """
        self.feed().maximize(verbose=verbose)

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
        return float(self.variables().get('logitdelta').value)

    @logitdelta.setter
    def logitdelta(self, v):
        self.variables().get('logitdelta').value = v

    @property
    def logscale(self):
        return float(self.variables().get('logscale').value)

    @logscale.setter
    def logscale(self, v):
        self.variables().get('logscale').value = v

    def mean(self):
        r"""Mean of the prior.

        Returns
        -------
        array_like
            :math:`\mathrm X\boldsymbol\beta`.
        """
        return dot(self._X, self.beta)

    @property
    def scale(self):
        r"""Overall variance.

        Returns
        -------
        float
            :math:`s`.
        """
        return exp(self.logscale)

    @scale.setter
    def scale(self, v):
        self.logscale = log(v)

    def set_variable_bounds(self, var_name, bounds):
        self.variables().get(var_name).bounds = bounds

    def unfix(self, var_name):
        r"""Let a variable be adjusted.

        Parameters
        ----------
        var_name : str
            Variable name.
        """
        Function.unfix(self, _to_internal_name(var_name))

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


def _to_internal_name(name):
    translation = dict(scale='logscale', delta='logitdelta', beta='beta')
    return translation[name]
