from __future__ import division

from numpy import exp
from numpy import log

from optimix import Function
from optimix import Scalar


class EyeCov(Function):
    r"""Identity covariance.

    .. math::

        s \mathrm I
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
        r"""Covariance matrix.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.
        """
        return self.scale * (x0 == x1)

    def derivative_logscale(self, x0, x1):
        r"""Covariance matrix derivative over :math:`\log s`.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.
        """
        return self.value(x0, x1)
