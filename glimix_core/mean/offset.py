from numpy import full, ones

from optimix import Func, Scalar


class OffsetMean(Func):
    r"""
    Offset mean function.

    It represents a mean vector

        o⋅𝟏

    of size n. The offset is given by the parameter o.

    Example
    -------

    .. doctest::

        >>> from glimix_core.mean import OffsetMean
        >>>
        >>> mean = OffsetMean()
        >>> mean.offset = 2.0
        >>> x = [0, 1, 2]
        >>> print(mean.value(x))
        [2. 2. 2.]
        >>> print(mean.gradient(x))
        {'offset': array([1., 1., 1.])}
        >>> print(mean)
        OffsetMean()
          offset: 2.0
        >>> mean.name = "mean-name"
        >>> print(mean)
        OffsetMean(): mean-name
          offset: 2.0
    """

    def __init__(self, n):
        self._offset = Scalar(0.0)
        self._offset.bounds = (-200.0, +200)
        Func.__init__(self, offset=self._offset)

    def value(self):
        """
        Offset mean.

        Returns
        -------
        M : (n,) ndarray
            o⋅𝟏.
        """
        return full(self._n, self._offset.value)

    def gradient(self):
        """
        Gradient of the offset function.

        Returns
        -------
        offset : (n,) ndarray
            𝟏.
        """
        return dict(offset=ones(self._n))

    @property
    def offset(self):
        """
        Offset parameter.
        """
        return self._offset.value

    @offset.setter
    def offset(self, v):
        self._offset.value = v

    def __str__(self):
        tname = type(self).__name__
        msg = "{}()".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  offset: {}".format(self.offset)
        return msg
