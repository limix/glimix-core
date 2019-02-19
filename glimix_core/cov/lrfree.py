from __future__ import division

from numpy import dot, ones, stack, zeros_like, asarray

from optimix import Function, Vector

from ..util.classes import NamedClass


class LRFreeFormCov(NamedClass, Function):
    r""" General semi-definite positive matrix of low rank.

    The covariance matrix K is given by LLᵗ, where L is a m×n matrix and m≥n. Therefore,
    K will have rank(A) ≤ n.

    Parameters
    ----------
    m : int
        Covariance dimension.
    n : int
        Upper limit of the covariance matrix rank.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import FreeFormCov
        >>>
        >>> cov = FreeFormCov(2)
        >>> print(cov.value([0, 1], [0, 1]))
        [[1. 1.]
         [1. 2.]]
        >>> print(cov.L)
        [[1. 0.]
         [1. 1.]]
        >>> print(cov.Lu)
        [1. 1. 1.]
        >>> g = cov.gradient([0, 1], [0, 1])
        >>> print(g['Lu'].shape)
        (2, 2, 3)
        >>> print(g['Lu'])
        [[[2. 0. 0.]
          [1. 1. 0.]]
        <BLANKLINE>
         [[1. 1. 0.]
          [0. 2. 2.]]]
        >>> cov.Lu[1] = -2
        >>> print(cov.L)
        [[ 1.  0.]
         [-2.  1.]]
        >>> print(cov.value([0, 1], [0, 1]))
        [[ 1. -2.]
         [-2.  5.]]
        >>> print(cov)
        FreeFormCov()
          Lu: [ 1. -2.  1.]
        >>> cov.name = "covname"
        >>> print(cov)
        FreeFormCov(): covname
          Lu: [ 1. -2.  1.]
    """

    def __init__(self, m, n):
        self._L = ones((m, n))
        Function.__init__(self, Lu=Vector(self._L.ravel()))
        NamedClass.__init__(self)

    @property
    def L(self):
        """ Matrix L from K=LLᵗ. """
        return self._L

    @L.setter
    def L(self, value):
        # self._L[:] = value
        self.Lu = asarray(value, float).ravel()

    @property
    def Lu(self):
        """ Matrix L in flat form."""
        return self.variables().get("Lu").value

    @Lu.setter
    def Lu(self, value):
        self.variables().get("Lu").value = value

    def value(self, x0, x1):
        r""" Covariance function evaluated at ``(x0, x1)``.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        array_like
            Submatrix of LLᵗ, row and column-indexed by x₀ and x₁.
        """
        return dot(self.L, self.L.T)[x0, ...][..., x1]

    def gradient(self, x0, x1):
        r""" Derivative of the covariance function evaluated at ``(x0, x1)``.

        Derivative over L.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        dict
            Dictionary having the `Lu` key for the derivative of LLᵗ, row and
            column-indexed by x₀ and x₁.
        """
        L = self.L
        Lo = zeros_like(L)
        grad = []
        for ii in range(self._L.shape[0] * self._L.shape[1]):
            row = ii // self._L.shape[1]
            col = ii % self._L.shape[1]
            Lo[row, col] = 1
            grad.append(dot(Lo, L.T) + dot(L, Lo.T))
            Lo[row, col] = 0

        grad = [g[x0, ...][..., x1] for g in grad]
        return dict(Lu=stack(grad, axis=-1))

    def __str__(self):
        tname = type(self).__name__
        msg = "{}()".format(tname)
        if self.name is not None:
            msg += ": {}".format(self.name)
        msg += "\n"
        msg += "  Lu: {}".format(self.Lu)
        return msg
