from numpy import asarray, ascontiguousarray, zeros

from optimix import Function, Vector


class LinearMean(Function):
    """
    Linear mean function, X𝜶.

    It defines X𝜶, for which X is a n×m matrix provided by the user and 𝜶 is a vector
    of size m.

    Example
    -------

    .. doctest::

        >>> from glimix_core.mean import LinearMean
        >>>
        >>> mean = LinearMean(2)
        >>> mean.effsizes = [1.0, -1.0]
        >>> mean.X = [[1.5, 0.2], [0.5, 0.4]]
        >>> print(mean.value())
        [1.3 0.1]
        >>> print(mean.gradient()["effsizes"])
        [[1.5 0.2]
         [0.5 0.4]]
        >>> mean.name = "𝐦"
        >>> print(mean)
        LinearMean(m=2): 𝐦
          effsizes: [ 1. -1.]
    """

    def __init__(self, m):
        """
        Constructor.

        Parameters
        ----------
        size : int
            Size m of 𝜶.
        """
        self._effsizes = Vector(zeros(m))
        self._effsizes.bounds = [(-200.0, +200)] * m
        self._X = None
        Function.__init__(self, "LinearMean", effsizes=self._effsizes)

    @property
    def X(self):
        """
        An n×m matrix of covariates.
        """
        return self._X

    @X.setter
    def X(self, X):
        self._X = asarray(X, float)

    def value(self):
        """
        Linear mean function.

        Returns
        -------
        𝐦 : (n,) ndarray
            X𝜶.
        """
        return self._X @ self._effsizes

    def gradient(self):
        """
        Gradient of the linear mean function over the effect sizes.

        Returns
        -------
        effsizes : (n, m) ndarray
            X.
        """
        return dict(effsizes=self._X)

    @property
    def effsizes(self):
        """
        Effect-sizes parameter, 𝜶, of size m.
        """
        return self._effsizes.value

    @effsizes.setter
    def effsizes(self, v):
        self._effsizes.value = ascontiguousarray(v)

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(m={})".format(tname, len(self.effsizes))
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  effsizes: {}".format(self.effsizes)
        return msg
