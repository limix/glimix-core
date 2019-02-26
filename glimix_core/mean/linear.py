from numpy import ascontiguousarray, zeros

from optimix import Func, Vector


class LinearMean(Func):
    """
    Linear mean function.

    It defines Xα, for which X is a n×m matrix provided by the user and α is a vector
    of size m.

    Parameters
    ----------
    size : int
        Size m of α.

    Example
    -------

    .. doctest::

        >>> from numpy import array
        >>> from glimix_core.mean import LinearMean
        >>>
        >>> mean = LinearMean(2)
        >>> mean.effsizes = [1.0, -1.0]
        >>> mean.X = array([[1.5, 0.2], [0.5, 0.4]])
        >>> print(mean.value())
        [1.3 0.1]
        >>> print(mean.gradient()["effsizes"])
        [[1.5 0.2]
         [0.5 0.4]]
        >>> mean.name = "M"
        >>> print(mean)
        LinearMean(m=2): M
          effsizes: [ 1. -1.]
    """

    def __init__(self, m):
        self._effsizes = Vector(zeros(m))
        self._effsizes.bounds = [(-200.0, +200)] * m
        self._X = None
        Func.__init__(self, "LinearMean", effsizes=self._effsizes)

    @property
    def X(self):
        """
        An n×m matrix of covariates.
        """
        return self._X

    @X.setter
    def X(self, X):
        self._X = X

    def value(self):
        """
        Linear mean function.

        Returns
        -------
        M : (n,) ndarray
            Xα.
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
        Effect-sizes parameter, α, of size m.
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
