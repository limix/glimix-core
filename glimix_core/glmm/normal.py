from __future__ import absolute_import, division, unicode_literals

from copy import copy

from liknorm import LikNormMachine
from numpy import asarray, dot, exp, log, pi, zeros
from numpy.linalg import slogdet, solve

from numpy_sugar.linalg import ddot, sum2diag
from optimix import Function, Scalar, Vector


class GLMMNormal(Function):
    r"""LMM with heterogeneous Normal likelihoods.

    Here we model

    .. math::

        \tilde{\boldsymbol\mu} \sim \mathcal N(\mathrm X\boldsymbol\beta,
        v_0 \mathrm K + v_1 \mathrm I + \tilde{\Sigma})

    where :math:`\tilde{\boldsymbol\mu}` and :math:`\tilde{\Sigma}` are
    typically given by EP approximation to non-normal likelihood (please, refer
    to :class:`glimix_core.glmm.expfam.GLMMExpFam`).
    Note that :math:`\tilde{\Sigma}` is a diagonal matrix with non-negative
    values.
    Those EP parameters are given to this class via their natural forms: ``eta``
    and ``tau``.

    Parameters
    ----------
    eta : array_like
        :math:`\tilde{\mu}_i \tilde{\sigma}^{-2}_i`.
    tau : array_like
        :math:`\tilde{\sigma}^{-2}_i`.
    X : array_like
        Covariates.
    QS : tuple
        Economic eigen decomposition of :math:`\mathrm K`.
    """

    def __init__(self, eta, tau, X, QS):
        Function.__init__(
            self,
            beta=Vector(zeros(X.shape[1])),
            logscale=Scalar(0.0),
            logitdelta=Scalar(0.0))

        self._eta = eta
        self._tau = tau
        self._X = X
        self._QS = QS
        self.set_nodata()
        self._factr = 1e5
        self._pgtol = 1e-6
        # self.variables().get('beta').listen(self.set_update_approx)
        # self.variables().get('logscale').listen(self.set_update_approx)
        # self.variables().get('logitdelta').listen(self.set_update_approx)

    def __copy__(self):
        gef = GLMMExpFam(self._y, self._lik_name, self._X, self._QS)

        d = gef.variables()
        s = self.variables()

        d.get('beta').value = asarray(s.get('beta').value, float)
        d.get('beta').bounds = s.get('beta').bounds

        for v in ['logscale', 'logitdelta']:
            d.get(v).value = float(s.get(v).value)
            d.get(v).bounds = s.get(v).bounds

        return gef

    @property
    def beta(self):
        return asarray(self.variables().get('beta').value, float)

    @beta.setter
    def beta(self, v):
        self.variables().get('beta').value = v
        # self.set_update_approx()

    def covariance(self):
        scale = exp(self.logscale)
        delta = 1 / (1 + exp(-self.logitdelta))
        return dict(QS=self._QS, scale=scale, delta=delta)

    def fix(self, var_name):
        Function.fix(self, var_name)
        # self.set_update_approx()

    def gradient(self):
        pass
        # g = self._ep.lml_derivatives(self._X)
        # ed = exp(-self.logitdelta)
        # es = exp(self.logscale)
        #
        # grad = dict()
        # grad['logitdelta'] = g['delta'] * (ed / (1 + ed)) / (1 + ed)
        # grad['logscale'] = g['scale'] * es
        # grad['beta'] = g['mean']
        #
        # return grad

    @property
    def logitdelta(self):
        return float(self.variables().get('logitdelta').value)

    @logitdelta.setter
    def logitdelta(self, v):
        self.variables().get('logitdelta').value = v
        # self.set_update_approx()

    @property
    def logscale(self):
        return float(self.variables().get('logscale').value)

    @logscale.setter
    def logscale(self, v):
        self.variables().get('logscale').value = v
        # self.set_update_approx()

    def mean(self):
        return dot(self._X, self.beta)

    def set_variable_bounds(self, var_name, bounds):
        self.variables().get(var_name).bounds = bounds

    def unfix(self, var_name):
        Function.unfix(self, var_name)

    def value(self):
        r"""Log of the marginal likelihood.

        Formally,

        .. math::

            \frac{n}{2}\log{2\pi} + \frac{1}{2} \log{\left|
                v_0 \mathrm K + v_1 \mathrm I + \tilde{\Sigma} \right|}
                    + \frac{1}{2}
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
        scale = exp(self.logscale)
        delta = 1 / (1 + exp(-self.logitdelta))

        v0 = scale * (1 - delta)
        v1 = scale * delta

        Q0 = self._QS[0][0]
        S0 = self._QS[1]

        mu = self._eta / self._tau
        K = dot(ddot(Q0, S0), Q0.T)

        n = K.shape[0]
        A = sum2diag(sum2diag(v0 * K, v1), 1 / self._tau)
        m = mu - self.mean()
        v = (n / 2) * log(2 * pi)
        v += (1 / 2) * slogdet(A)[1]
        v += (1 / 2) * dot(m, solve(A, m))

        return v
