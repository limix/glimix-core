from __future__ import division

from numpy import asarray, dot, ones, zeros, zeros_like

from optimix import Function, Vector

from .._util import format_function, format_named_arr


class LRFreeFormCov(Function):
    """
    General semi-definite positive matrix of low rank, K = LLᵗ.

    The covariance matrix K is given by LLᵗ, where L is a n×m matrix and n≥m. Therefore,
    K will have rank(K) ≤ m.

    Parameters
    ----------
    n : int
        Covariance dimension.
    m : int
        Upper limit of the covariance matrix rank.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import LRFreeFormCov
        >>> cov = LRFreeFormCov(3, 2)
        >>> print(cov.L)
        [[1. 1.]
        [1. 1.]
        [1. 1.]]
        >>> cov.L = [[1, 2], [0, 3], [1, 3]]
        >>> print(cov.L)
        [[1. 2.]
         [0. 3.]
         [1. 3.]]
        >>> cov.name = "F"
        >>> print(cov)
        LRFreeFormCov(n=3, m=2): F
        L: [[1. 2.]
            [0. 3.]
            [1. 3.]]
    """

    def __init__(self, n, m):
        self._L = ones((n, m))
        self._Lu = Vector(self._L.ravel())
        Function.__init__(self, "LRFreeFormCov", Lu=self._Lu)

    @property
    def Lu(self):
        """
        Lower-triangular, flat part of L.
        """
        return self._Lu.value

    @Lu.setter
    def Lu(self, v):
        self._Lu.value = v

    @property
    def L(self):
        """
        Matrix L from K=LLᵗ.

        Returns
        -------
        L : (n, m) ndarray
            Parametric matrix.
        """
        return self._L

    @L.setter
    def L(self, value):
        self._Lu.value = asarray(value, float).ravel()

    def value(self):
        """
        Covariance matrix.

        Returns
        -------
        ndarray : (n, n)
            K = LLᵀ.
        """
        return dot(self.L, self.L.T)

    def gradient(self):
        """
        Derivative of the covariance matrix over L.

        Derivative of K over the lower triangular, flat part of L:

            ∂K/∂Lᵢⱼ = ALᵀ + LAᵀ,

        where Aᵢⱼ is an n×m matrix of zeros except at [Aᵢⱼ]ᵢⱼ=1.

        Returns
        -------
        Lu : ndarray
            Derivative of K over the lower-triangular, flat part of L.
        """
        L = self.L
        n = self.L.shape[0]
        Lo = zeros_like(L)
        grad = {"Lu": zeros((n, n, n * self._L.shape[1]))}
        for ii in range(self._L.shape[0] * self._L.shape[1]):
            row = ii // self._L.shape[1]
            col = ii % self._L.shape[1]
            Lo[row, col] = 1
            grad["Lu"][..., ii] = dot(Lo, L.T) + dot(L, Lo.T)
            Lo[row, col] = 0

        return grad

    def __str__(self):
        L = self._L
        msg = format_function(self, n=L.shape[0], m=L.shape[1]) + "\n"
        msg += format_named_arr("L", L)
        return msg
