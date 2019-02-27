from numpy import add

from optimix import Function


class SumCov(Function):
    """
    Sum of covariance functions, K = K₀ + K₁ + ⋯.

    The mathematical representation is K = K₀ + K₁ + ⋯.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import LinearCov, SumCov
        >>> from numpy.random import RandomState
        >>>
        >>> random = RandomState(0)
        >>> cov_left = LinearCov()
        >>> cov_right = LinearCov()
        >>>
        >>> cov_left.scale = 0.5
        >>> X0 = random.randn(4, 20)
        >>> X1 = random.randn(4, 15)
        >>>
        >>> cov_left.X = X0
        >>> cov_right.X = X1
        >>>
        >>> cov = SumCov([cov_left, cov_right])
        >>> cov_left.name = "A"
        >>> cov_right.name = "B"
        >>> cov.name = "A+B"
        >>> print(cov)
        SumCov(covariances=...): A+B
          LinearCov(): A
            scale: 0.5
          LinearCov(): B
            scale: 1.0
    """

    def __init__(self, covariances):
        self._covariances = [c for c in covariances]
        Function.__init__(self, "SumCov", composite=self._covariances)

    def value(self):
        r"""
        Sum of covariance matrices.

        Returns
        -------
        dict
            :math:`f_0 + f_1 + \dots`
        """
        return add.reduce([cov.value() for cov in self._covariances])

    def gradient(self):
        r"""
        Sum of covariance function derivatives.

        Returns
        -------
        dict
            :math:`f_0' + f_1' + \dots`
        """
        grad = {}
        for i, f in enumerate(self._covariances):
            for varname, g in f.gradient().items():
                grad[f"{self._name}[{i}].{varname}"] = g
        return grad

    def __str__(self):
        tname = type(self).__name__
        msg = "{}(covariances=...)".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        for c in self._covariances:
            spl = str(c).split("\n")
            msg = msg + "\n" + "\n".join(["  " + s for s in spl])
        return msg
