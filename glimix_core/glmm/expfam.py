from __future__ import absolute_import, division, unicode_literals

from copy import copy

from liknorm import LikNormMachine
from numpy import asarray, dot, exp, zeros

from optimix import Function, Scalar, Vector

from ..ep import EPLinearKernel


class GLMMExpFam(Function):
    def __init__(self, y, lik_name, X, QS):
        Function.__init__(
            self,
            beta=Vector(zeros(X.shape[1])),
            logscale=Scalar(0.0),
            logitdelta=Scalar(0.0))

        self._y = y
        self._lik_name = lik_name
        self._X = X
        self._QS = QS
        self._ep = EPLinearKernel(X.shape[0])
        self._ep.set_compute_moments(self.compute_moments)
        self._machine = LikNormMachine(lik_name, 1000)
        self.set_nodata()
        self.update_approx = True
        self.variables().get('beta').listen(self.set_update_approx)
        self.variables().get('logscale').listen(self.set_update_approx)
        self.variables().get('logitdelta').listen(self.set_update_approx)

    def __copy__(self):
        gef = GLMMExpFam(self._y, self._lik_name, self._X, self._QS)
        gef.__dict__['_ep'] = copy(self._ep)
        gef.__dict__['_ep'].set_compute_moments(gef.compute_moments)
        gef.update_approx = self.update_approx

        beta = gef.variables().get('beta')
        beta.value = asarray(self.variables().get('beta').value, float)
        beta.bounds = self.variables().get('beta').bounds

        logscale = gef.variables().get('logscale')
        logscale.value = float(self.variables().get('logscale').value)
        logscale.bounds = self.variables().get('logscale').bounds

        logitdelta = gef.variables().get('logitdelta')
        logitdelta.value = float(self.variables().get('logitdelta').value)
        logitdelta.bounds = self.variables().get('logitdelta').bounds

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
        return asarray(self.variables().get('beta').value, float)

    @beta.setter
    def beta(self, v):
        self.variables().get('beta').value = v
        self.set_update_approx()

    def compute_moments(self, eta, tau, moments):
        self._machine.moments(self._y, eta, tau, moments)

    def copy(self):
        pass

    def covariance(self):
        scale = exp(self.logscale)
        delta = 1 / (1 + exp(-self.logitdelta))
        return dict(QS=self._QS, scale=scale, delta=delta)

    def fix(self, var_name):
        Function.fix(self, var_name)
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
        return float(self.variables().get('logitdelta').value)

    @logitdelta.setter
    def logitdelta(self, v):
        self.variables().get('logitdelta').value = v
        self.set_update_approx()

    @property
    def logscale(self):
        return float(self.variables().get('logscale').value)

    @logscale.setter
    def logscale(self, v):
        self.variables().get('logscale').value = v
        self.set_update_approx()

    def mean(self):
        return dot(self._X, self.beta)

    def set_update_approx(self, _=None):
        self.update_approx = True

    def set_variable_bounds(self, var_name, bounds):
        self.variables().get(var_name).bounds = bounds
        self.set_update_approx()

    def unfix(self, var_name):
        Function.unfix(self, var_name)
        self.set_update_approx()

    def value(self):
        self._update_approx()
        return self._ep.lml()
