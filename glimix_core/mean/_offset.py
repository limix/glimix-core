from numpy import full, ones
from optimix import Function, Scalar


class OffsetMean(Function):
    r"""
    Offset mean function, θ⋅𝟏.

    It represents a mean vector 𝐦 = θ⋅𝟏 of size n. The offset is given by the parameter
    θ.

    Example
    -------

    .. doctest::

        >>> from glimix_core.mean import OffsetMean
        >>>
        >>> mean = OffsetMean(3)
        >>> mean.offset = 2.0
        >>> print(mean.value())
        [2. 2. 2.]
        >>> print(mean.gradient())
        {'offset': array([1., 1., 1.])}
        >>> mean.name = "𝐦"
        >>> print(mean)
        OffsetMean(): 𝐦
          offset: 2.0
    """

    def __init__(self, n):
        """
        Constructor.

        Parameters
        ----------
        n : int
            Size of the 𝟏 array.
        """
        self._offset = Scalar(0.0)
        self._offset.bounds = (-200.0, +200)
        self._n = n
        Function.__init__(self, "OffsetMean", offset=self._offset)

    def fix_offset(self):
        """
        Prevent θ update during optimization.
        """
        self._fix("offset")

    def unfix_offset(self):
        """
        Enable θ update during optimization.
        """
        self._unfix("offset")

    def value(self):
        """
        Offset mean.

        Returns
        -------
        𝐦 : (n,) ndarray
            θ⋅𝟏.
        """
        return full(self._n, self._offset.value)

    def gradient(self):
        """
        Gradient of the offset function.

        Returns
        -------
        offset : (n,) ndarray
            Vector 𝟏.
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
