from __future__ import division

from numpy import asarray, exp, log, newaxis, atleast_1d

from optimix import Function, Scalar


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
            :math:`s \delta[\mathrm x_0 = \mathrm x_1]`.
        """
        x0 = asarray(x0)
        x1 = asarray(x1)
        x0_ = atleast_1d(x0).ravel()[:, newaxis]
        x1_ = atleast_1d(x1).ravel()[newaxis, :]
        v = self.scale * (x0_ == x1_)
        return v.reshape(x0.shape + x1.shape)

    def gradient(self, x0, x1):
        r"""Derivative of the covariance function evaluated at `(x0, x1)`.

        Derivative of the covariance function over :math:`\log(s)`.

        Args:
            x0 (array_like): left-hand side sample or samples.
            x1 (array_like): right-hand side sample or samples.

        Returns:
            :math:`s \delta[\mathrm x_0 = \mathrm x_1]`.
        """
        return dict(logscale=self.value(x0, x1))
