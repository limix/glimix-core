from __future__ import division

from numpy import exp, log
from numpy_sugar import epsilon

from optimix import Function, Scalar


class GivenCov(Function):
    r"""Given covariance function.

    The mathematical representation is

    .. math::

        f(x_0, x_1) = s \mathrm K_{x_0, x_1},

    where :math:`s` is the scale parameter and :math:`\mathrm K` is given.
    In other words, the user specify passes a covariance matrix, which is
    then indexed by the ``x_0`` and ``x_1``.
    """

    def __init__(self, K):
        Function.__init__(self, logscale=Scalar(0.0))
        self._K = K

    @property
    def scale(self):
        r"""Scale parameter."""
        return exp(self.variables().get("logscale").value)

    @scale.setter
    def scale(self, scale):
        scale = max(scale, epsilon.tiny)
        self.variables().get("logscale").value = log(scale)

    def value(self, x0, x1):
        r"""Covariance function evaluated at ``(x0, x1)``.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        array_like
            Submatrix of :math:`s \mathrm K`, row and column-indexed by
            ``x0`` and ``x1``.
        """
        return self.scale * self._K[x0, :][..., x1]

    def gradient(self, x0, x1):
        r"""Derivative of the covariance function evaluated at ``(x0, x1)``.

        Derivative of the covariance function over :math:`\log(s)`.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        dict
            Dictionary having the `logscale` key for the derivative, row and
            column-indexed by ``x0`` and ``x1``.
        """
        return dict(logscale=self.scale * self._K[x0, :][..., x1])
