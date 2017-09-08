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
# class NGLMM(Function):
#     def __init__(self, y, X, QS):
#         super(GLMM, self).__init__(X.shape[0])
#         Function.__init__(
#             self,
#             beta=Vector(zeros(X.shape[1])),
#             logscale=Scalar(0.0),
#             logitdelta=Scalar(0.0))
#
#         self._factr = 1e5
#         self._pgtol = 1e-5
#
#         self._logger = logging.getLogger(__name__)
#
#         logscale = self.variables()['logscale']
#         logscale.bounds = (log(1e-3), 7.)
#         logscale.listen(self._set_need_prior_update)
#
#         logitdelta = self.variables()['logitdelta']
#         logitdelta.bounds = (-inf, +inf)
#         logitdelta.listen(self._set_need_prior_update)
#
#         self.variables()['beta'].listen(self._set_need_prior_update)
#
#         self._y = y
#
#         if not isinstance(QS, tuple):
#             raise ValueError("QS must be a tuple.")
#
#         if not isinstance(QS[0], tuple):
#             raise ValueError("QS[0] must be a tuple.")
#
#         self._QS = QS
#
#         self.set_nodata()
#
#     @property
#     def likelihood_means(self):
#         return self._y[0]
#
#     @property
#     def likelihood_variances(self):
#         return self._y[1]
#
#     def mean(self):
#         return dot(self._X, self.beta)
#
#     def fix(self, var_name):
#         if var_name == 'scale':
#             Function.fix(self, 'logscale')
#         elif var_name == 'delta':
#             Function.fix(self, 'logitdelta')
#         elif var_name == 'beta':
#             Function.fix(self, 'beta')
#         else:
#             raise ValueError("Unknown parameter name %s." % var_name)
#         self._set_need_prior_update()
#
#     def unfix(self, var_name):
#         if var_name == 'scale':
#             Function.unfix(self, 'logscale')
#         elif var_name == 'delta':
#             Function.unfix(self, 'logitdelta')
#         elif var_name == 'beta':
#             Function.unfix(self, 'beta')
#         else:
#             raise ValueError("Unknown parameter name %s." % var_name)
#         self._set_need_prior_update()
#
#     @property
#     def scale(self):
#         return float(exp(self.variables().get('logscale').value))
#
#     @scale.setter
#     def scale(self, v):
#         self.variables().get('logscale').value = log(v)
#
#     @property
#     def delta(self):
#         return float(1 / (1 + exp(-self.variables().get('logitdelta').value)))
#
#     @delta.setter
#     def delta(self, v):
#         v = clip(v, epsilon.large, 1 - epsilon.large)
#         self.variables().get('logitdelta').value = log(v / (1 - v))
#
#     @property
#     def beta(self):
#         return asarray(self.variables().get('beta').value, float)
#
#     @beta.setter
#     def beta(self, v):
#         self.variables().get('beta').value = v
#
#     def value(self):
#         s = self.scale
#         delta = self.delta
#         beta = self.beta
#
#         A = s((1 - delta) * K + delta * I) + S
#
#         # det(2 PI A)^{-1/2} * exp( -((mu - m)inv(A)(mu - m))/2 )
#         pass
#
#     # def gradient(self):
#     #     v = self.variables().get('logitdelta').value
#     #     x = self.variables().get('logscale').value
#     #     g = self._lml_derivatives(self._X)
#     #     ev = exp(-v)
#     #     grad = [
#     #         g['delta'] * (ev / (1 + ev)) / (1 + ev), g['scale'] * exp(x),
#     #         g['mean']
#     #     ]
#     #
#     #     return grad
