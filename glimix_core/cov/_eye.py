from numpy import exp, eye, log
from optimix import Function, Scalar

from .._util import format_function


class EyeCov(Function):
    """
    Identity covariance function, K = s·I.

    The parameter s is the scale of the matrix.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import EyeCov
        >>>
        >>> cov = EyeCov(2)
        >>> cov.scale = 2.5
        >>> print(cov.value())
        [[2.5 0. ]
         [0.  2.5]]
        >>> g = cov.gradient()
        >>> print(g['logscale'])
        [[2.5 0. ]
         [0.  2.5]]
        >>> cov.name = "I"
        >>> print(cov)
        EyeCov(dim=2): I
          scale: 2.5

    Parameters
    ----------
    dim : int
        Matrix dimension, d.
    """

    def __init__(self, dim):
        """
        Constructor.

        Parameters
        ----------
        dim : int
            Matrix dimension, d.
        """
        self._dim = dim
        self._I = eye(dim)
        self._logscale = Scalar(0.0)
        Function.__init__(self, "EyeCov", logscale=self._logscale)
        self._logscale.bounds = (-20.0, +10)

    @property
    def scale(self):
        """
        Scale parameter.
        """
        return exp(self._logscale)

    @scale.setter
    def scale(self, scale):
        from numpy_sugar import epsilon

        scale = max(scale, epsilon.tiny)
        self._logscale.value = log(scale)

    @property
    def dim(self):
        """
        Dimension of the matrix, d.

        It corresponds to the number of rows and to the number of columns.
        """
        return self._I.shape[0]

    def value(self):
        """
        Covariance matrix.

        Returns
        -------
        K : ndarray
            s⋅I, for scale s and a d×d identity matrix I.
        """
        return self.scale * self._I

    def gradient(self):
        """
        Derivative of the covariance matrix over log(s), s⋅I.

        Returns
        -------
        logscale : ndarray
            s⋅I, for scale s and a d×d identity matrix I.
        """
        return dict(logscale=self.value())

    def __str__(self):
        return format_function(self, {"dim": self._I.shape[0]}, [("scale", self.scale)])
