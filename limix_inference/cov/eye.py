from __future__ import division

from numpy import exp
from numpy import log
from numpy import eye
from numpy import ix_
from numpy import isscalar
from numpy import ascontiguousarray
from numpy import atleast_1d

from optimix import Function
from optimix import Scalar


class EyeCov(Function):
    r"""Identity covariance function.

    The mathematical representation is

    .. math::

        f(\mathrm x_0, \mathrm x_1) = s \delta[\mathrm x_0 = \mathrm x_1]

    where :math:`s` is the scale parameter and :math:`\delta` is the Kronecker
    delta.
    """
    def __init__(self):
        Function.__init__(self, logscale=Scalar(0.0))

    @property
    def scale(self):
        r"""Scale parameter."""
        return exp(self.get('logscale'))

    @scale.setter
    def scale(self, scale):
        self.set('logscale', log(scale))

    def value(self, x0, x1):
        r"""Covariance function evaluated at `(x0, x1)`.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.

        Returns:
            :math:`s \delta[\mathrm x_0 = \mathrm x_1]`.
        """
        if isscalar(x0) and isscalar(x1):
            return self.scale * (x0 == x1)

        one_scalar = isscalar(x0) or isscalar(x1)

        x0 = ascontiguousarray(atleast_1d(x0), int)
        x1 = ascontiguousarray(atleast_1d(x1), int)

        x0 = x0.ravel()
        x1 = x1.ravel()

        I = eye(len(x0), len(x1))
        I = I[ix_(x0, x1)]
        I *= self.scale

        if one_scalar:
            return I.ravel()

        return I


    def derivative_logscale(self, x0, x1):
        r"""Derivative of the covariance function evaluated at `(x0, x1)`.

        Derivative of the covariance function over :math:`\log(s)`.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.

        Returns:
            :math:`s \delta[\mathrm x_0 = \mathrm x_1]`.
        """
        return self.value(x0, x1)
