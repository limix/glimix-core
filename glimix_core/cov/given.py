from __future__ import division

from numpy import exp, log

from optimix import Func, Scalar


class GivenCov(Func):
    r"""Given covariance function.

    The covariance matrix is the provided matrix K₀ scaled by s: K = s⋅K₀.

    Parameters
    ----------
    K0 : array_like
        A semi-definite positive matrix.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import GivenCov
        >>> from numpy import dot
        >>> from numpy.random import RandomState
        >>>
        >>> G = RandomState(0).randn(5, 3)
        >>> K0 = dot(G, G.T)
        >>> cov = GivenCov(K0)
        >>> cov.scale = 1.3
        >>> cov.name = "K"
        >>> print(cov)
        GivenCov(K=...): K
          scale: 1.3
    """

    def __init__(self, K0):
        self._logscale = Scalar(0.0)
        Func.__init__(self, "GivenCov", logscale=self._logscale)
        self._logscale.bounds = (-20.0, +10)
        self._K0 = K0

    @property
    def scale(self):
        """
        Scale parameter, s.
        """
        return exp(self._logscale.value)

    @scale.setter
    def scale(self, scale):
        from numpy_sugar import epsilon

        scale = max(scale, epsilon.tiny)
        self._logscale.value = log(scale)

    def value(self):
        """
        Covariance matrix, s⋅K₀.

        Returns
        -------
        array_like
            s⋅K₀.
        """
        return self.scale * self._K0

    def gradient(self):
        r"""
        Derivative of the covariance matrix over log(s).

        Returns
        -------
        logscale
            s⋅K₀.
        """
        return dict(logscale=self.scale * self._K0)

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(K=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  scale: {}".format(self.scale)
        return msg
