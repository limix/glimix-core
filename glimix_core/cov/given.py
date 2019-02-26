from __future__ import division

from numpy import exp, log

from optimix import Func, Scalar


class GivenCov(Func):
    r"""Given covariance function.

    The mathematical representation is

    .. math::

        f(x_0, x_1) = s \mathrm K_{x_0, x_1},

    where :math:`s` is the scale parameter and :math:`\mathrm K` is given.
    In other words, the user specify passes a covariance matrix, which is
    then indexed by the ``x_0`` and ``x_1``.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import GivenCov
        >>> from numpy import dot
        >>> from numpy.random import RandomState
        >>>
        >>> K = RandomState(0).randn(5, 5)
        >>> K = dot(K, K.T)
        >>> cov = GivenCov(K)
        >>> cov.scale = 1.3
        >>> print(cov)
        GivenCov(K=...)
          scale: 1.3
        >>> cov.name = "covname"
        >>> print(cov)
        GivenCov(K=...): covname
          scale: 1.3
    """

    def __init__(self, K):
        self._logscale = Scalar(0.0)
        Func.__init__(self, "GivenCov", logscale=self._logscale)
        self._logscale.bounds = (-20.0, +10)
        self._K = K

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
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        array_like
            Submatrix of :math:`s \mathrm K`, row and column-indexed by
            ``x0`` and ``x1``.
        """
        return self.scale * self._K

    def gradient(self):
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
        return dict(logscale=self.scale * self._K)

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(K=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  scale: {}".format(self.scale)
        return msg
