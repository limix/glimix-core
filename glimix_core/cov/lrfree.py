from __future__ import division

from numpy import asarray, dot, ones, stack, zeros_like

from optimix import Function, Vector

from ..util.classes import NamedClass
from ..util import format_function, format_named_arr


class LRFreeFormCov(NamedClass, Function):
    """
    General semi-definite positive matrix of low rank.

    The covariance matrix K is given by LLᵗ, where L is a m×n matrix and m≥n. Therefore,
    K will have rank(K) ≤ n.

    Parameters
    ----------
    m : int
        Covariance dimension.
    n : int
        Upper limit of the covariance matrix rank.

    Example
    -------

    .. doctest::

        >>> from glimix_core.cov import LRFreeFormCov
        >>> cov = LRFreeFormCov(3, 2)
        >>> cov.L
        array([[1., 1.],
            [1., 1.],
            [1., 1.]])

        >>> cov.L = [[1, 2], [0, 3], [1, 3]]
        >>> cov.L
        array([[1., 2.],
            [0., 3.],
            [1., 3.]])
        >>> cov.gradient([0, 1, 2], [0, 1, 2])
        {'Lu': array([[[2., 4., 0., 0., 0., 0.],
                [0., 3., 1., 2., 0., 0.],
                [1., 3., 0., 0., 1., 2.]],
        <BLANKLINE>
            [[0., 3., 1., 2., 0., 0.],
                [0., 0., 0., 6., 0., 0.],
                [0., 0., 1., 3., 0., 3.]],
        <BLANKLINE>
            [[1., 3., 0., 0., 1., 2.],
                [0., 0., 1., 3., 0., 3.],
                [0., 0., 0., 0., 2., 6.]]])}
        >>> cov.gradient([0, 1, 2], [0, 1, 2])["Lu"].shape
        (3, 3, 6)
        >>> print(cov)
        LRFreeFormCov(m=3, n=2)
        L: [[1. 2.]
            [0. 3.]
            [1. 3.]]
        >>> cov.name = "covname"
        >>> print(cov)
        LRFreeFormCov(m=3, n=2): covname
        L: [[1. 2.]
            [0. 3.]
            [1. 3.]]
    """

    def __init__(self, m, n):
        self._L = ones((m, n))
        Function.__init__(self, Lu=Vector(self._L.ravel()))
        NamedClass.__init__(self)

    @property
    def L(self):
        """
        Matrix L from K=LLᵗ.

        Returns
        -------
        L : (m, n) ndarray
            Parametric matrix.
        """
        return self._L

    @L.setter
    def L(self, value):
        self.variables().get("Lu").value = asarray(value, float).ravel()

    def value(self, x0, x1):
        """
        Covariance function evaluated at (x₀,x₁).

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        ndarray
            Submatrix of K, row and column-indexed by x₀ and x₁.
        """
        return dot(self.L, self.L.T)[x0, ...][..., x1]

    def gradient(self, x0, x1):
        """
        Derivative of the covariance function evaluated at (x₀,x₁).

        Derivative of K over L.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        Lu : ndarray
            Derivative of K over the flattened L, row and column-indexed by x₀ and x₁.
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
        L = self._L
        msg = format_function(self, m=L.shape[0], n=L.shape[1]) + "\n"
        msg += format_named_arr("L", L)
        return msg
