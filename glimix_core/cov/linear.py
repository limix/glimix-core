from __future__ import division

from numpy import exp, log

from optimix import Func, Scalar


class LinearCov(Func):
    r"""Linear covariance function.

    The mathematical representation is

    .. math::

        f(\mathrm x_0, \mathrm x_1) = s \mathrm x_0^\intercal \mathrm x_1,

    where :math:`s` is the scale parameter.
    """

    def __init__(self):
        self._logscale = Scalar(0.0)
        self._X = None
        Func.__init__(self, "LinearCov", logscale=self._logscale)
        self._logscale.bounds = (-20.0, +10)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    @property
    def scale(self):
        r"""Scale parameter."""
        return exp(self._logscale.value)

    @scale.setter
    def scale(self, scale):
        from numpy_sugar import epsilon

        scale = max(scale, epsilon.tiny)
        self._logscale.value = log(scale)

    def value(self):
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
        X = self.X
        return self.scale * (X @ X.T)

    def gradient(self):
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
        return dict(logscale=self.value())

    def __str__(self):
        tname = type(self).__name__
        msg = "{}()".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  scale: {}".format(self.scale)
        return msg
