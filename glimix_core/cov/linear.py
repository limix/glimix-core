from __future__ import division

from numpy import exp, log, stack

from optimix import Function, Scalar

from ..util.classes import NamedClass


class LinearCov(NamedClass, Function):
    r"""Linear covariance function.

    The mathematical representation is

    .. math::

        f(\mathrm x_0, \mathrm x_1) = s \mathrm x_0^\intercal \mathrm x_1,

    where :math:`s` is the scale parameter.
    """

    def __init__(self):
        Function.__init__(self, logscale=Scalar(0.0))
        self.variables().get("logscale").bounds = (-20.0, +10)
        NamedClass.__init__(self)

    @property
    def scale(self):
        r"""Scale parameter."""
        return exp(self.variables().get("logscale").value)

    @scale.setter
    def scale(self, scale):
        from numpy_sugar import epsilon

        scale = max(scale, epsilon.tiny)
        self.variables().get("logscale").value = log(scale)

    def value(self, x0, x1):
        r"""Covariance function evaluated at ``(x0, x1)``.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample or samples.
        x1 : array_like
            Right-hand side sample or samples.

        Returns
        -------
        array_like
            :math:`s \mathrm x_0^\intercal \mathrm x_1`.
        """
        x0 = stack(x0, axis=0)
        x1 = stack(x1, axis=0)
        return self.scale * x0.dot(x1.T)

    def gradient(self, x0, x1):
        r"""Derivative of the covariance function evaluated at ``(x0, x1)``.

        Derivative of the covariance function over :math:`\log(s)`:

        .. math::

            s \mathrm x_0^\intercal \mathrm x_1.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample or samples.
        x1 : array_like
            Right-hand side sample or samples.

        Returns
        -------
        dict
            Dictionary having the `logscale` key for the derivative.
        """
        x0 = stack(x0, axis=0)
        x1 = stack(x1, axis=0)
        return dict(logscale=self.scale * x0.dot(x1.T))

    def __str__(self):
        tname = type(self).__name__
        msg = "{}()".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  scale: {}".format(self.scale)
        return msg
