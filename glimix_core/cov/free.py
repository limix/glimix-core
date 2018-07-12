from __future__ import division

from numpy import dot, ones, stack, tril_indices_from, zeros, zeros_like

from optimix import Function, Vector

from ..util.classes import NamedClass


class FreeFormCov(NamedClass, Function):
    r"""General semi-definite positive matrix.

    A :math:`d`-by-:math:`d` covariance matrix :math:`\mathrm K` will have
    ``((d + 1) * d) / 2`` parameters defining the lower triangular part of
    the Cholesky matrix ``L``: :math:`\mathrm L\mathrm L^\intercal=\mathrm K`.

    Parameters
    ----------
    size : int
        Dimension :math:`d` of the free-form covariance matrix.

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

    def __init__(self, size):
        size = int(size)
        tsize = ((size + 1) * size) // 2
        self._L = zeros((size, size))
        self._tril = tril_indices_from(self._L)
        self._L[self._tril] = 1
        Function.__init__(self, Lu=Vector(ones(tsize)))
        NamedClass.__init__(self)

    @property
    def L(self):
        """Cholesky decomposition of the covariance matrix."""
        self._L[self._tril] = self.variables().get("Lu").value
        return self._L

    @L.setter
    def L(self, value):
        self._L[:] = value
        self.variables().get("Lu").value = self._L[self._tril]

    @property
    def Lu(self):
        """Cholesky decomposition of the covariance matrix in flat form."""
        return self.variables().get("Lu").value

    @Lu.setter
    def Lu(self, value):
        self.variables().get("Lu").value = value

    def value(self, x0, x1):
        r"""Covariance function evaluated at ``(x0, x1)``.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        array_like
            Submatrix of :math:`\mathrm L\mathrm L^\intercal`, row and
            column-indexed by ``x0`` and ``x1``.
        """
        return dot(self.L, self.L.T)[x0, ...][..., x1]

    def gradient(self, x0, x1):
        r"""Derivative of the covariance function evaluated at ``(x0, x1)``.

        Derivative over the lower triangular part of :math:`\mathrm L`.

        Parameters
        ----------
        x0 : array_like
            Left-hand side sample indices.
        x1 : array_like
            Right-hand side sample indices.

        Returns
        -------
        dict
            Dictionary having the `Lu` key for the lower triangular part
            of the derivative of :math:`\mathrm L\mathrm L^\intercal`, row and
            column-indexed by ``x0`` and ``x1``.
        """
        L = self.L
        Lo = zeros_like(L)
        grad = []
        for ii in range(len(self._tril[0])):
            row = self._tril[0][ii]
            col = self._tril[1][ii]
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
