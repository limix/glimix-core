from __future__ import division

from numpy import exp, log

from optimix import Function, Scalar


class GivenCov(Function):
    r"""Given covariance function.

    The mathematical representation is

    .. math::

        f(x_0, x_1) = s \mathrm K_{x0, x1}

    where :math:`s` is the scale parameter and :math:`\mathrm K` is given.
    """

    def __init__(self, K):
        Function.__init__(self, logscale=Scalar(0.0))
        self._K = K

    @property
    def scale(self):
        r"""Scale parameter."""
        return exp(self.variables().get('logscale').value)

    @scale.setter
    def scale(self, scale):
        self.variables().get('logscale').value = log(scale)

    def value(self, x0, x1):
        r"""Covariance function evaluated at `(x0, x1)`.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.

        Returns:
            :math:`s \mathrm x_0^\intercal \mathrm x_1`.
        """
        return self.scale * self._K[x0, :][..., x1]

    def gradient(self, x0, x1):
        r"""Derivative of the covariance function evaluated at `(x0, x1)`.

        Derivative of the covariance function over :math:`\log(s)`.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.

        Returns:
            :math:`s \mathrm x_0^\intercal \mathrm x_1`.
        """
        return dict(logscale=self.scale * self._K[x0, :][..., x1])
