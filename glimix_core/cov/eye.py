from numpy import exp, log, eye

from optimix import Func, Scalar


class EyeCov(Func):
    r"""Identity covariance function.

    The mathematical representation is

    .. math::

        f(\mathrm x_0, \mathrm x_1) = s \delta[\mathrm x_0 = \mathrm x_1]

    where :math:`s` is the scale parameter and :math:`\delta` is the Kronecker
    delta.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import EyeCov
        >>>
        >>> cov = EyeCov()
        >>> cov.scale = 2.5
        >>>
        >>> item0 = 0
        >>> item1 = 1
        >>> print(cov.value(item0, item1))
        0.0
        >>> g = cov.gradient(item0, item1)
        >>> print(g['logscale'])
        0.0
        >>> item0 = [0, 1, 2]
        >>> item1 = [0, 1, 2]
        >>> print(cov.value(item0, item1))
        [[2.5 0.  0. ]
         [0.  2.5 0. ]
         [0.  0.  2.5]]
        >>> g = cov.gradient(item0, item1)
        >>> print(g['logscale'])
        [[2.5 0.  0. ]
         [0.  2.5 0. ]
         [0.  0.  2.5]]
        >>> print(cov)
        EyeCov()
          scale: 2.5
        >>> cov.name = "identity"
        >>> print(cov)
        EyeCov(): identity
          scale: 2.5
    """

    def __init__(self):
        self._logscale = Scalar(0.0)
        Func.__init__(self, "EyeCov", logscale=self._logscale)
        self._logscale.bounds = (-20.0, +10)
        self._I = None

    @property
    def scale(self):
        r"""Scale parameter."""
        return exp(self._logscale)

    @scale.setter
    def scale(self, scale):
        from numpy_sugar import epsilon

        scale = max(scale, epsilon.tiny)
        self._logscale.value = log(scale)

    @property
    def dim(self):
        return self._I.shape[0]

    @dim.setter
    def dim(self, dim):
        self._I = eye(dim)

    def value(self):
        r"""Covariance function evaluated at `(x0, x1)`.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample or samples.
        x1 : array_like
            Right-hand side sample or samples.

        Returns
        -------
        array_like
            :math:`s \delta[\mathrm x_0 = \mathrm x_1]`.
        """
        return self.scale * self._I

    def gradient(self):
        r"""Derivative of the covariance function evaluated at `(x0, x1)`.

        Derivative of the covariance function over :math:`\log(s)`.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample or samples.
        x1 : array_like
            Right-hand side sample or samples.

        Returns
        -------
        dict
            Dictionary having the `logscale` key for
            :math:`s \delta[\mathrm x_0 = \mathrm x_1]`.
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
