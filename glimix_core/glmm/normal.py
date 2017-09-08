# from __future__ import absolute_import, division, unicode_literals
#
# import logging
#
# from liknorm import LikNormMachine
# from numpy import asarray, clip, dot, exp, inf, log, zeros
# from numpy.linalg import LinAlgError
# from numpy_sugar import epsilon
#
# from optimix import Function, Scalar, Vector
#
# from ..ep import EPLinearKernel
#
#
# class GLMMExpFam(Function):
#     def __init__(self, y, lik_name, X, QS):
#         # super(GLMMExpFam, self).__init__(X.shape[0])
#         Function.__init__(
#             self,
#             beta=Vector(zeros(X.shape[1])),
#             logscale=Scalar(0.0),
#             logitdelta=Scalar(0.0))
#
#         self._ep = EPLinearKernel()
#         self._machine = LikNormMachine(lik_name, 1000)
#         self.set_nodata()
#         self.update_posterior = True
#
#         # self._factr = 1e5
#         # self._pgtol = 1e-5
#
#         # logscale = self.variables()['logscale']
#         # logscale.bounds = (log(1e-3), 7.)
#         # logscale.listen(self.set_update_posterior)
#         #
#         # logitdelta = self.variables()['logitdelta']
#         # logitdelta.bounds = (-inf, +inf)
#         # logitdelta.listen(self.set_update_posterior)
#         #
#         # self.variables()['beta'].listen(self.set_update_posterior)
#         #
#         # if lik_name.lower() == 'poisson':
#         #     y = clip(y, 0, 25000)
#         #
#         # if isinstance(y, list):
#         #     y = tuple(y)
#         # elif not isinstance(y, tuple):
#         #     y = (y, )
#         #
#         # self._y = tuple([asarray(i, float) for i in y])
#         # self._X = X
#         #
#         # if not isinstance(QS, tuple):
#         #     raise ValueError("QS must be a tuple.")
#         #
#         # if not isinstance(QS[0], tuple):
#         #     raise ValueError("QS[0] must be a tuple.")
#         #
#         # self._QS = QS
#         #
#         # self._lik_name = lik_name
#
#         # self._need_prior_update = True
#
#     def copy(self):
#         glmm = GLMM(self._y, self._lik_name, self._X, self._QS)
#
#         glmm.scale = self.scale
#         glmm.delta = self.delta
#         glmm.beta = self.beta
#
#         self._copy_to(glmm)
#         glmm.need_prior_update = self._need_prior_update
#
#         return glmm
#
#     def set_update_posterior(self, _=None):
#         self.update_posterior = True
#
#     def __del__(self):
#         if hasattr(self, '_machine'):
#             self._machine.finish()
#
#     def _compute_moments(self):
#         tau = self._cav['tau']
#         eta = self._cav['eta']
#         self._machine.moments(self._y, eta, tau, self._moments)
#
#     def fix(self, var_name):
#         Function.fix(self, var_name)
#         self.set_update_posterior()
#
#     def unfix(self, var_name):
#         Function.unfix(self, var_name)
#         self.set_update_posterior()
#
#     @property
#     def logscale(self):
#         return float(self.variables().get('logscale').value)
#
#     @scale.setter
#     def logscale(self, v):
#         self.variables().get('logscale').value = v
#         self.set_update_posterior()
#
#     @property
#     def logitdelta(self):
#         return float(self.variables().get('logitdelta').value)
#
#     @delta.setter
#     def logitdelta(self, v):
#         self.variables().get('logitdelta').value = v
#         self.set_update_posterior()
#
#     @property
#     def beta(self):
#         return asarray(self.variables().get('beta').value, float)
#
#     @beta.setter
#     def beta(self, v):
#         self.variables().get('beta').value = v
#         self.set_update_posterior()
#
#     def mean(self):
#         pass
#
#     def cov(self):
#         return dict(QS=self._QS, scale=self.scale, delta=self.delta)
#
#     def _update_posterior(self):
#         if not self.update_posterior:
#             return
#
#         self._set_prior(self.mean(), self.cov())
#         self.update_posterior = False
#
#     def value(self):
#         self._update_posterior()
#         return self._lml()
#
#     def gradient(self):
#
#         x = self.variables().get('logscale').value
#         g = self._lml_derivatives(self._X)
#         ed = exp(-self.logitdelta)
#         es = exp(self.logscale)
#         grad = [
#             g['delta'] * (ed / (1 + ed)) / (1 + ed), g['scale'] * es, g['mean']
#         ]
#
#         return dict(logitdelta=grad[0], logscale=grad[1], beta=grad[2])
