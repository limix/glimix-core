from __future__ import division

from numpy import exp
from numpy import log

from optimix import Scalar
from optimix import Function


class LinearCov(Function):
    r"""Linear covariance function.

    The mathematical representation is

    .. math::

        f(\mathrm x_0, \mathrm x_1) = s \mathrm x_0^\intercal \mathrm x_1

    where :math:`s` is the scale parameter.
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
            :math:`s \mathrm x_0^\intercal \mathrm x_1`.
        """
        return self.scale * x0.dot(x1.T)

    def derivative_logscale(self, x0, x1):
        r"""Derivative of the covariance function evaluated at `(x0, x1)`.

        Derivative of the covariance function over :math:`\log(s)`.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.

        Returns:
            :math:`s \mathrm x_0^\intercal \mathrm x_1`.
        """
        return self.scale * x0.dot(x1.T)
